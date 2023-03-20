import scanpy as sc
import anndata as ad
import random
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import pandas as pd
from collections import Counter
import numpy as np
import torch
import config

def read_all_adata(data_dir = config.data_dir):
    E_MTAB_6149_19tumor_addmeta_10X_file = data_dir + "E_MTAB_6149_19tumor_addmeta.h5ad"
    GSE131907_58ALL_LUAD_addmeta_10X_file = data_dir + "GSE131907_58ALL_LUAD_addmeta.h5ad"
    GSE136831_78All_IPF_COPD_CTL_addmeta_10X_file = data_dir + "GSE136831_78All_IPF_COPD_CTL_addmeta.h5ad"
    E_MTAB_8221_8fetal_addmeta_10X_file = data_dir + "E_MTAB_8221_8fetal_addmeta.h5ad"
    GSE128169_13ALL_SSC_addmeta_10X_file = data_dir + "GSE128169_13ALL_SSC_addmeta.h5ad"
    E_MTAB_6653_12tumor_addmeta_10X_file = data_dir + "E_MTAB_6653_12tumor_addmeta.h5ad"
    GSE128033_18ALL_IPF_addmeta_10X_file = data_dir + "GSE128033_18ALL_IPF_addmeta.h5ad"
    lungAtlasSmartSeq2_file = data_dir + "adata_HLCA_ss2_8545_count.anno.h5ad"
    lungAtlas10X_file = data_dir + "adata_HLCA_10X_60993_count.anno.h5ad"

    print("loading data" , flush=True)

    E_MTAB_6149_19tumor_addmeta_10X = sc.read(E_MTAB_6149_19tumor_addmeta_10X_file)
    GSE131907_58ALL_LUAD_addmeta_10X = sc.read(GSE131907_58ALL_LUAD_addmeta_10X_file)
    GSE136831_78All_IPF_COPD_CTL_addmeta_10X = sc.read(GSE136831_78All_IPF_COPD_CTL_addmeta_10X_file)
    E_MTAB_8221_8fetal_addmeta_10X = sc.read(E_MTAB_8221_8fetal_addmeta_10X_file)
    GSE128169_13ALL_SSC_addmeta_10X = sc.read(GSE128169_13ALL_SSC_addmeta_10X_file)
    E_MTAB_6653_12tumor_addmeta_10X = sc.read(E_MTAB_6653_12tumor_addmeta_10X_file)
    GSE128033_18ALL_IPF_addmeta_10X = sc.read(GSE128033_18ALL_IPF_addmeta_10X_file)
    adata_lungAtlasSmartSeq2 = sc.read(lungAtlasSmartSeq2_file)
    adata_lungAtlas10X = sc.read(lungAtlas10X_file)

    outer = ad.concat([E_MTAB_6149_19tumor_addmeta_10X, GSE131907_58ALL_LUAD_addmeta_10X, GSE136831_78All_IPF_COPD_CTL_addmeta_10X,
                    E_MTAB_8221_8fetal_addmeta_10X, GSE128169_13ALL_SSC_addmeta_10X, E_MTAB_6653_12tumor_addmeta_10X, GSE128033_18ALL_IPF_addmeta_10X, 
                    adata_lungAtlasSmartSeq2, adata_lungAtlas10X], join="inner")
    print("loaded data" , flush=True)
    return outer

def make_counter():
    adata = read_all_adata()
    gene_expression = adata.X
    gene_sums = csr_matrix.sum(gene_expression, axis=0)
    gene_sums_df = pd.DataFrame(gene_sums.T, columns=['expression_sum'], index=adata.var_names)
    gene_sums_dict = gene_sums_df.to_dict()['expression_sum']
    counter = Counter(gene_sums_dict)
    return counter


class scDataset(Dataset):
    def __init__(self, datadir = None, adata = None):
        super(scDataset, self).__init__()
        if datadir is not None:
            self.datadir = datadir
        else:
            self.datadir = "/home/xuguang/scEMD/data_backup/Seurat3_ntiss10x.anno.h5ad"
        if adata is not None:
            self.adata = adata
        else:
            self.adata = sc.read_h5ad(self.datadir)
        # self.adata = read_all_adata()
        self.length = self.adata.shape[0]
        self.vocab_size = self.adata.shape[1]
        self.gene_list = np.array(self.adata.var_names)
        self.padding_value = len(self.gene_list) + 1
        if 'cell_type' in self.adata.obs.columns:
            self.lable_dict, self.lables = np.unique(np.array(self.adata.obs['cell_type']), return_inverse=True)
        elif 'cell_type_combine' in self.adata.obs.columns:
            self.lable_dict, self.lables = np.unique(np.array(self.adata.obs['cell_type_combine']), return_inverse=True)
        else: 
            self.lables = np.array(self.adata.obs['leiden'],dtype=int)
            self.lable_dict = list(range(57))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        gene_indexs_bool = (self.adata.X[idx,] != 0).todense().A[0]
        gene_indexs = np.array(range(len(gene_indexs_bool)))[gene_indexs_bool]
        gene_exprs = self.adata.X[idx,gene_indexs_bool].todense().A[0]
        sort_index = np.argsort(-gene_exprs) # 从大到小排列
        return gene_indexs[sort_index], gene_exprs[sort_index], self.lables[idx]


