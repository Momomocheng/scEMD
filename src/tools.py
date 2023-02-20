from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch
import math

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



