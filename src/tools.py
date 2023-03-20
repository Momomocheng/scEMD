from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch
import math
import scanpy as sc
import scipy.sparse as sp
import anndata
import numpy as np
import pandas as pd

def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
      optimizer (:class:`~torch.optim.Optimizer`):
        The optimizer for which to schedule the learning rate.
      num_warmup_steps (:obj:`int`):
        The number of steps for the warmup phase.
      num_training_steps (:obj:`int`):
        The total number of training steps.
      num_cycles (:obj:`float`, `optional`, defaults to 0.5):
        The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
        following a half-cosine).
      last_epoch (:obj:`int`, `optional`, defaults to -1):
        The index of the last epoch when resuming training.

    Return:
      :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# 处理输入数据，返回加了mask_token_id和pad_token_id的gene_indexs_masked，都使用pad_token_id填充的gene_index_label
def data_mask(gene_indexs, pad_index, mask_prob, mask_token_id, pad_token_id, device):
    prob = torch.full(gene_indexs.shape, mask_prob)
    mask_bool = torch.bernoulli(prob).bool().to(device)
    mask_bool[pad_index] = False
    gene_indexs_masked = gene_indexs.masked_fill(mask_bool, mask_token_id)
    gene_index_label = gene_indexs.masked_fill(~mask_bool, pad_token_id)
    return gene_indexs_masked, mask_bool, gene_index_label




import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))

class NpairLoss(nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchor, positive, target):
        batch_size = anchor.size(0)
        target = target.view(target.size(0), 1)

        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        loss_ce = cross_entropy(logit, target)
        l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size

        loss = loss_ce + self.l2_reg*l2_loss*0.25
        return loss

def convert_adata_genes(adata_src, adata_target):
    """
    将adata_src 在基因上向adata_target对齐, 缺失部分使用0填充
    Parameters
    ----------
    adata_src : anndata.AnnData
        Source annotated data matrix.
    adata_target : anndata.AnnData
        Target annotated data matrix.

    Returns
    -------
    anndata.AnnData
        Source annotated data matrix with gene names matching those of the target matrix.
    """
    # 获取目标数据集的基因名称
    target_genes = adata_target.var_names
    # 获取源数据集的基因名称
    src_genes = adata_src.var_names
    # 找到源数据集和目标数据集中共有的基因
    common_genes = np.intersect1d(src_genes, target_genes)
    # 找到目标数据集中有但是源数据集中没有的基因
    diff_genes = np.array(list(set(target_genes) - set(common_genes)))
    # 创建一个全为0的稀疏矩阵
    n_cells = adata_src.n_obs
    n_genes = len(diff_genes)
    zero_X = sp.csr_matrix((n_cells, n_genes), dtype=np.float32)
    # 按列拼接稀疏矩阵
    new_X = sp.hstack([adata_src[:, common_genes].X, zero_X])
    # 获取新的基因名称
    new_genes = np.concatenate([common_genes, diff_genes])
    # 创建一个新的 AnnData 对象
    obs = adata_src.obs.copy()
    var = pd.DataFrame(index = new_genes)
    layers = {"counts": new_X}
    adata_new = anndata.AnnData(new_X, obs, var, layers=layers)
    adata_new = adata_new[:,target_genes]
    myarray = adata_new.X.toarray()
    X = sp.csr_matrix(myarray)
    adata_new = anndata.AnnData(X, obs, pd.DataFrame(index = target_genes), layers=layers)
    return adata_new