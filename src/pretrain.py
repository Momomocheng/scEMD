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

from scDataset import *
from torchsampler import ImbalancedDatasetSampler
from torch.optim import AdamW

from tools import *
from model import scEMD




parser = argparse.ArgumentParser()
parser.add_argument("--gene_num", type=int, default=26485, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=42, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=1, help='Number of batch size.')
parser.add_argument("--n_workers", type=int, default=32, help='Number of dataloader workers.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
# parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
# parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--mask_prob", type=float, default=0.15, help='Probability of masking.')
parser.add_argument("--replace_prob", type=float, default=0.9, help='Probability of replacing with [MASK] token for masking.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='/home/xuguang/scEMD/data_backup/adata_HLCA_10X_60993_count.anno.h5ad', help='Path of data for pretraining.')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='HLCA_10X_pretrain', help='Pretrained model name.')
parser.add_argument("--maxlength", type=str, default=200, help='max input length.')

args = parser.parse_args([])

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
CLASS = args.gene_num + 2
MASK_PROB = args.mask_prob
REPLACE_PROB = args.replace_prob
RANDOM_TOKEN_PROB = 0.
MASK_TOKEN_ID = CLASS - 1
PAD_TOKEN_ID = CLASS - 2
# MASK_IGNORE_TOKEN_IDS = [26485]
POS_EMBED_USING = args.pos_embed

N_WORKERS = args.n_workers
MAX_LENGTH = args.maxlength

model_name = args.model_name
ckpt_dir = args.ckpt_dir


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return out_gene_indexs[:max_length], out_gene_exprs[:max_length], torch.FloatTensor(cell_lables).long()[:max_length], pad_index[:max_length]

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    # shuffle=True,
    drop_last=True,
    num_workers=N_WORKERS,
    pin_memory=True,
    collate_fn=collate_batch,
    persistent_workers=True,
    sampler=ImbalancedDatasetSampler(train_dataset),
)
valid_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=N_WORKERS,
    drop_last=True,
    pin_memory=True,
    collate_fn=collate_batch,
    persistent_workers=True,
)

model = scEMD(d_model=15, n_labels=len(dataset.lable_dict), vocab_size=CLASS,
              embedding_dim = 15, dim_feedforward = 10, nhead=1, num_layers=2)

model.to(device)


optimizer = AdamW(model.parameters(), lr=1e-3)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = 1000, num_training_steps = EPOCHS * dataset.length / BATCH_SIZE)
loss_fn = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN_ID, reduction='mean')
# progeress bar
console = Console()
live = Live(console=console)
live.start()

for i in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    epoch_loss = 0.0
    for index, data in enumerate(train_loader):
        index += 1
        gene_indexs, gene_exprs, cell_lables, pad_index = data
        gene_indexs, gene_exprs, cell_lables, pad_index = gene_indexs.to(device), gene_exprs.to(device), cell_lables.to(device), pad_index.to(device)
        gene_indexs_masked, mask_bool, gene_index_label = data_mask(gene_indexs, pad_index = pad_index, mask_prob = MASK_PROB, mask_token_id = MASK_TOKEN_ID, pad_token_id = PAD_TOKEN_ID, device = device)
        gene_logits, cell_logits = model(gene_indexs_masked, gene_exprs, padding_mask = pad_index | mask_bool)
        # mask gene predict
        gene_loss = loss_fn(gene_logits.view(gene_logits.shape[0]*gene_logits.shape[1],-1), gene_indexs.view(-1))
        gene_accuracy = torch.mean((gene_logits.argmax(dim=-1) == gene_index_label).float())
        # cell predict
        cell_loss = loss_fn(cell_logits, cell_lables)
        cell_accuracy = torch.mean((cell_logits.argmax(dim=-1) == gene_index_label).float())
        # Updata model
        loss = gene_loss + cell_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        live.update(f"EPOCH:{i}/{EPOCHS}, batch:{index}:\n epoch_loss:{epoch_loss:.4f}, loss:{loss:.4f}\n gene_accuracy:{gene_accuracy:.4f}, gene_loss:{gene_loss:.4f} \n cell_accuracy:{cell_accuracy:.4f}, cell_loss:{cell_loss:.4f}")
    
    epoch_loss = running_loss / index

live.stop()









