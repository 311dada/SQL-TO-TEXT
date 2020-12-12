"""
    ALL LAYERS MODULE
        * Position Encoding Layer
        * Positionwise Feed Forward Layer
        * RAT Layer
        * GCN Layer
"""
import torch.nn as nn
from Model.attention import RelationMultiHeadAttention
import torch
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """
    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        self.pe = self.pe.to(emb.device)
        if step is None:
            if self.pe.size(0) < emb.size(1):
                raise ValueError(
                    f"Sequence is {emb.size(1)} but PositionalEncoding is"
                    f" limited to {self.pe.size(0)}. See max_len argument.")
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """Layer definition.
        Args:
            x: ``(batch_size, input_len, model_dim)``
        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """

        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


# RAT layer
class RATLayer(nn.Module):
    def __init__(self,
                 head_num,
                 d_model,
                 d_ff,
                 d_rel,
                 dropout,
                 rel_share=True,
                 k_v_share=True,
                 mode="concat"):
        super(RATLayer, self).__init__()

        self.RAT = RelationMultiHeadAttention(head_num,
                                              d_model,
                                              d_rel,
                                              rel_share=rel_share,
                                              k_v_share=k_v_share,
                                              mode=mode)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, r_k=None, r_v=None, mask=None):
        input_norm = self.layer_norm(x)
        context = self.RAT(input_norm, input_norm, input_norm, r_k, r_v, mask)
        out = self.drop(context) + x
        return self.feed_forward(out)
