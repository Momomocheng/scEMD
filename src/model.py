from torch.nn.modules.transformer import *
from torch.nn.modules.transformer import _get_activation_fn, _get_clones

class TransformerEncoderLayer_Expr_Attention(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer_Expr_Attention, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, kdim=1)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_Expr_Attention, self).__setstate__(state)

    def forward(self, src: Tensor, src_expr: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attn_output_avg_weights = self.self_attn(src, src_expr, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder_Expr_Attention(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder_Expr_Attention, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, src_expr, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_expr=src_expr, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

class scEMD(nn.Module):
    def __init__(self, d_model, n_labels, vocab_size, embedding_dim, dim_feedforward, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        # embedding for scEMD
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim)
        # Encoder Part
        self.encoder_layer = TransformerEncoderLayer_Expr_Attention(
            d_model=d_model, dim_feedforward=dim_feedforward, nhead=nhead, dropout=dropout)
        self.encoder = TransformerEncoder_Expr_Attention(self.encoder_layer, num_layers = num_layers)

        # cell classify part
        self.classifier = nn.Linear(d_model, n_labels)

    def forward(self, gene_indexs, gene_exprs, padding_mask, return_gene_encoded = False):
        """
        args:
            gene_indexs: (batch size, length)
            gene_exprs: (batch size, length)
            mask: (batch size, length)
            padding_mask: (batch size, length)
        return:
          out: (batch size, n_spks)
        """
        # gene_exprs: (batch size, length, embedding_dim)
        # gene_exprs = gene_exprs.repeat(self.embedding_dim, 1, 1).transpose(0, 1).transpose(1, 2)
        gene_exprs = gene_exprs.unsqueeze(-1)
        # gene_exprs: (length, batch size, embedding_dim)
        gene_exprs = gene_exprs.permute(1, 0, 2)
        # gene_embeddings: (batch size, length, embedding_dim)
        gene_embeddings = self.embedding(gene_indexs)
        # gene_embeddings: (length, batch size, embedding_dim)
        gene_embeddings = gene_embeddings.permute(1, 0, 2)
        # gene_encoded: (length, batch size, d_model)
        gene_encoded = self.encoder(gene_embeddings, gene_exprs,
                                    src_key_padding_mask = padding_mask)
        # gene_encoded: (batch size, length, d_model)
        gene_encoded = gene_encoded.transpose(0, 1)

        # pred_gene_index: (batch size, length, vocab_size)
        pred_gene_index = gene_encoded @ self.embedding.weight.t()

        # predict cell type
        # average pooling
        # cell_encoded:(batch size, d_model)
        cell_encoded = torch.mean(gene_encoded, dim=1)
        pred_cell_labels = self.classifier(cell_encoded)

        return pred_gene_index, pred_cell_labels
    

