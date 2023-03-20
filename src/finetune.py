# -*- coding: utf-8 -*-

import os
import gc
import argparse
import json
import random
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
import scanpy as sc
import anndata as ad
from rich.console import Console
from rich.live import Live
import logging

from scDataset import *
# from torchsampler import ImbalancedDatasetSampler
from torch.optim import AdamW

from tools import *
from model import scEMD

parser = argparse.ArgumentParser()
parser.add_argument("--gene_num", type=int, default=26485, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=42, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=8, help='Number of batch size.')
parser.add_argument("--n_workers", type=int, default=32, help='Number of dataloader workers.')
parser.add_argument("--learning_rate", type=float, default=1e-3, help='Learning rate.')
# parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
# parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--mask_prob", type=float, default=0.15, help='Probability of masking.')
# parser.add_argument("--replace_prob", type=float, default=0.9, help='Probability of replacing with [MASK] token for masking.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='/home/xuguang/scEMD/data_backup/adata_HLCA_10X_60993_count.anno.h5ad', help='Path of data for pretraining.')
parser.add_argument("--file_path", type=str, default='./saved_model/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='HLCA_10X_Finetune4', help='Pretrained model name.')
parser.add_argument("--maxlength", type=str, default=1000, help='max input length.')
parser.add_argument("--model_path", type=str, 
                    default="/home/xuguang/scEMD/saved_model/Dataset_9_pretrain/ep30gene_accuracy_0.0001.pth", help='pretrained model path')

args = parser.parse_args([])

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
MASK_PROB = args.mask_prob
# REPLACE_PROB = args.replace_prob
# RANDOM_TOKEN_PROB = 0.
CLASS = args.gene_num + 2
MASK_TOKEN_ID = CLASS - 1
PAD_TOKEN_ID = CLASS - 2
POS_EMBED_USING = args.pos_embed
N_WORKERS = args.n_workers
MAX_LENGTH = args.maxlength
model_name = args.model_name
file_path = args.file_path

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# 创建一个StreamHandler，将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# 创建一个FileHandler，将日志写入文件
if not os.path.exists(file_path + model_name):
    os.mkdir(file_path + model_name)
file_handler = logging.FileHandler(file_path + model_name +'/mylog.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# 将Handler添加到Logger中
logger = logging.getLogger()
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info]: Use {device} now!")
# device = torch.device("cpu")

dataset = scDataset()
# 定义划分比例
train_ratio = 0.8
test_ratio = 1 - train_ratio
# 计算划分大小
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
# 使用 random_split 函数进行划分
train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

# collate_fn for DataLoader
def collate_batch(batch, padding_value=PAD_TOKEN_ID, max_length=MAX_LENGTH):
    gene_indexs, gene_exprs, cell_lables = zip(*batch)
    batch_num = len(gene_indexs)
    max_len = max([s.size for s in gene_indexs])
    out_dims = (batch_num, max_len)
    out_gene_indexs = torch.full(size = out_dims, fill_value = padding_value)
    out_gene_exprs = torch.full(size = out_dims, fill_value = 0.0)
    pad_index = torch.full(size = out_dims, fill_value = True)
    for i, tensor in enumerate(gene_indexs):
        length = tensor.size
        out_gene_indexs[i, :length, ...] = torch.from_numpy(tensor)
        out_gene_exprs[i, :length, ...] = torch.from_numpy(gene_exprs[i])
        pad_index[i, :length, ...] = False
    if(max_length<max_len):
        return out_gene_indexs[:,:max_length], out_gene_exprs[:,:max_length], torch.FloatTensor(cell_lables).long()[:max_length], pad_index[:,:max_length]
    else:
        return out_gene_indexs, out_gene_exprs, torch.FloatTensor(cell_lables).long(), pad_index

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    # shuffle=True,
    drop_last=True,
    num_workers=N_WORKERS,
    pin_memory=True,
    collate_fn=collate_batch,
    # persistent_workers=True,
    # sampler=ImbalancedDatasetSampler(train_dataset),
)
valid_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=N_WORKERS,
    drop_last=True,
    pin_memory=True,
    collate_fn=collate_batch,
    # persistent_workers=True,
)


adata_10X = sc.read("/home/xuguang/scEMD/data_backup/Seurat3_ntiss10x.anno.h5ad")
adata_GSE131907 = sc.read("/home/xuguang/scEMD/data_backup/adata_GSE131907_10X_139995Cells_count_subsetGene_aligned.h5ad")
adata_GSE136831 = sc.read("/home/xuguang/scEMD/data_backup/adata_GSE136831_10X_subset307169Cells_aligned.h5ad")
# adata_GSE131907 = sc.read("/home/xuguang/scEMD/data_backup/adata_GSE131907_10X_139995Cells_count_subsetGene.h5ad")
# adata_GSE136831 = sc.read("/home/xuguang/scEMD/data_backup/adata_GSE136831_10X_subset307169 Cells.h5ad")
print('loaded gse131907 gse136831!')
# adata_GSE131907 = convert_adata_genes(adata_GSE131907, adata_10X)
# adata_GSE136831 = convert_adata_genes(adata_GSE136831, adata_10X)
# print('converted gse131907 gse136831!')

def valid_ss2(model, device):
    adata_dir = "/home/xuguang/scEMD/data_backup/adata_HLCA_ss2_8545_count.anno.h5ad"
    adata = sc.read(adata_dir)
    adata = convert_adata_genes(adata, adata_10X)
    testloader = DataLoader(
        scDataset(adata = adata),
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch,
        # persistent_workers=True,
    )
    model.eval()
    running_accuracy = 0.0
    for index, data in enumerate(testloader):
        torch.cuda.empty_cache()
        index += 1
        gene_indexs, gene_exprs, cell_lables, pad_index = data
        gene_indexs, gene_exprs, cell_lables, pad_index = gene_indexs.to(device), gene_exprs.to(device), cell_lables.to(device), pad_index.to(device)
        # gene_indexs_masked, mask_bool, gene_index_label = data_mask(gene_indexs, pad_index = pad_index, mask_prob = MASK_PROB, mask_token_id = MASK_TOKEN_ID, pad_token_id = PAD_TOKEN_ID, device = device)
        _, cell_logits = model(gene_indexs, gene_exprs, padding_mask = pad_index)
        cell_accuracy = torch.mean((cell_logits.argmax(dim=-1) == cell_lables).float())
        running_accuracy += cell_accuracy
    return running_accuracy/index

def valid_gse(model, device, dataset_name):
    if(dataset_name == 'GSE131907'):
        adata = adata_GSE131907
    if(dataset_name == 'GSE136831'):
        adata = adata_GSE136831
    my_dataset = scDataset(adata = adata)
    testloader = DataLoader(
        my_dataset,
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch,
        # persistent_workers=True,
    )
    cell_types = dataset.lable_dict
    test_cell_types = my_dataset.lable_dict
    model.eval()
    running_accuracy = 0.0
    for index, data in enumerate(testloader):
        torch.cuda.empty_cache()
        index += 1
        gene_indexs, gene_exprs, cell_lables, pad_index = data
        gene_indexs, gene_exprs, cell_lables, pad_index = gene_indexs.to(device), gene_exprs.to(device), cell_lables.to(device), pad_index.to(device)
        # gene_indexs_masked, mask_bool, gene_index_label = data_mask(gene_indexs, pad_index = pad_index, mask_prob = MASK_PROB, mask_token_id = MASK_TOKEN_ID, pad_token_id = PAD_TOKEN_ID, device = device)
        _, cell_logits = model(gene_indexs, gene_exprs, padding_mask = pad_index)
        preds = cell_logits.argmax(1)
        pred_celltypes = [cell_types[pred] for pred in preds]
        true_celltypes = [test_cell_types[label] for label in cell_lables]
        accuracy = 0
        for i,pred_celltype in enumerate(pred_celltypes):
            if(pred_celltype in true_celltypes[i].split(',')):
                accuracy += 1
        # Compute accuracy.
        accuracy = torch.tensor(accuracy/len(preds)).float()
        running_accuracy += accuracy
    return running_accuracy/index

model = torch.load(args.model_path)
# model = scEMD(d_model=100, n_labels=len(dataset.lable_dict), vocab_size=CLASS,
#               embedding_dim = 100, dim_feedforward = 100, nhead=2, num_layers=2)
# model = nn.DataParallel(model,device_ids=[0,1])
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = 1000, num_training_steps = EPOCHS * dataset.length / BATCH_SIZE)
loss_fn = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN_ID, reduction='mean')
loss_npair = NpairLoss()

for i in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    epoch_loss = 0.0
    cell_accuracys = 0
    for index, data in enumerate(train_loader):
        torch.cuda.empty_cache()
        index += 1
        gene_indexs, gene_exprs, cell_lables, pad_index = data
        gene_indexs, gene_exprs, cell_lables, pad_index = gene_indexs.to(device), gene_exprs.to(device), cell_lables.to(device), pad_index.to(device)
        # gene_indexs_masked, mask_bool, gene_index_label = data_mask(gene_indexs, pad_index = pad_index, mask_prob = MASK_PROB, mask_token_id = MASK_TOKEN_ID, pad_token_id = PAD_TOKEN_ID, device = device)
        _, cell_logits = model(gene_indexs, gene_exprs, padding_mask = pad_index)
        # cell predict
        cell_loss = loss_fn(cell_logits, cell_lables)
        cell_accuracy = torch.mean((cell_logits.argmax(dim=-1) == cell_lables).float())
        # Updata model
        loss = cell_loss
        cell_accuracys += cell_accuracy
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_loss += loss.item()
    epoch_loss = running_loss / index
    logger.info(f"EPOCH:{i}/{EPOCHS}, batch:{index} epoch_loss:{epoch_loss:.4f}, loss:{loss:.4f} Training_cell_accuracy:{cell_accuracys/index:.4f}, cell_loss:{cell_loss:.4f}")
    #save model
    output_path = file_path + model_name + "/ep%d" % i + ".pth"
    torch.save(model.cpu(),output_path)
    model.to(device)

    # eval
    model.eval()
    eval_cell_accuracys = 0
    for index, data in enumerate(valid_loader):
        torch.cuda.empty_cache()
        index += 1
        gene_indexs, gene_exprs, cell_lables, pad_index = data
        gene_indexs, gene_exprs, cell_lables, pad_index = gene_indexs.to(device), gene_exprs.to(device), cell_lables.to(device), pad_index.to(device)
        _, cell_logits = model(gene_indexs, gene_exprs, padding_mask = pad_index)
        cell_accuracy = torch.mean((cell_logits.argmax(dim=-1) == cell_lables).float())
        eval_cell_accuracys += cell_accuracy
    logger.info(f"EPOCH:{i}/{EPOCHS}, Eval_cell_accuracy:{eval_cell_accuracys/index:.4f}")
    # test
    test_accuracy_ss2 = valid_ss2(model, device)
    test_accuracy_GSE131907 = valid_gse(model, device, 'GSE131907')
    test_accuracy_GSE136831 = valid_gse(model, device, 'GSE136831')
    logger.info(f"EPOCH:{i}/{EPOCHS}, Test_ss2_accuracy:{test_accuracy_ss2:.4f}, Test_GSE131907_accuracy:{test_accuracy_GSE131907:.4f}, Test_GSE136831_accuracy:{test_accuracy_GSE136831:.4f}")
