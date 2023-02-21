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

data_dir = config.data_dir
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

E_MTAB_6149_19tumor_addmeta_10X.obs["sample_name_for_split"] = E_MTAB_6149_19tumor_addmeta_10X.obs["sample"]
GSE131907_58ALL_LUAD_addmeta_10X.obs["sample_name_for_split"] = GSE131907_58ALL_LUAD_addmeta_10X.obs["sample_name"]
GSE136831_78All_IPF_COPD_CTL_addmeta_10X.obs["sample_name_for_split"] = GSE136831_78All_IPF_COPD_CTL_addmeta_10X.obs["sample_name"]
E_MTAB_8221_8fetal_addmeta_10X.obs["sample_name_for_split"] = E_MTAB_8221_8fetal_addmeta_10X.obs["sample_name"]
GSE128169_13ALL_SSC_addmeta_10X.obs["sample_name_for_split"] = GSE128169_13ALL_SSC_addmeta_10X.obs["sample"]
E_MTAB_6653_12tumor_addmeta_10X.obs["sample_name_for_split"] = E_MTAB_6653_12tumor_addmeta_10X.obs["sample_name"]
GSE128033_18ALL_IPF_addmeta_10X.obs["sample_name_for_split"] = GSE128033_18ALL_IPF_addmeta_10X.obs["sample"]
adata_lungAtlasSmartSeq2.obs["sample_name_for_split"] = adata_lungAtlasSmartSeq2.obs["sample"]
adata_lungAtlas10X.obs["sample_name_for_split"] = adata_lungAtlas10X.obs["sample"]

outer = ad.concat([E_MTAB_6149_19tumor_addmeta_10X, GSE131907_58ALL_LUAD_addmeta_10X, GSE136831_78All_IPF_COPD_CTL_addmeta_10X,
                    E_MTAB_8221_8fetal_addmeta_10X, GSE128169_13ALL_SSC_addmeta_10X, E_MTAB_6653_12tumor_addmeta_10X, GSE128033_18ALL_IPF_addmeta_10X, 
                    adata_lungAtlasSmartSeq2, adata_lungAtlas10X], join="outer")
print("loaded data" , flush=True)

# 按照sample列进行拆分
groups = outer.obs["sample_name_for_split"].unique()
adata_list = []
for g in groups:
    adata_list.append(outer[outer.obs["sample_name_for_split"] == g])
print("拆分!" , flush=True)

def calculate_leiden_clusters(adata):
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    if not "log1p" in adata.uns_keys():
        sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata)
    return list(adata.obs['leiden'])

leiden_list = []
for i,adata in enumerate(adata_list):
    print(i)
    leiden_list = leiden_list + calculate_leiden_clusters(adata)
print(outer.shape[0])
print(len(leiden_list))

outer.obs['leiden'] = leiden_list

outer.write("/home/xuguang/scEMD/data/adata_For_Pretrain.h5ad")