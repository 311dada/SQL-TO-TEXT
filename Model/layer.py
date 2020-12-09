"""
    ALL LAYERS MODULE
        * Positionwise Feed Forward Layer
        * RAT Layer
        * GCN Layer
"""
import torch.nn as nn
from Model.attention import RelationMultiHeadAttention


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
