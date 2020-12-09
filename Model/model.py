"""
    ALL MODEL
        * RGT
        * Transformer
            * with Relative Position Self Attention
            * with Absolute Position Self Attention
            * with Graph Attention solely
        * GCN
        * BiLSTM
        * TreeLSTM
"""
import torch
import torch.nn as nn
from Model.encoder import RGTEncoder, RATEncoder
from Model.decoder import LSTMDecoder, SimpleLSTMDecoder
from Model.utils import max_pooling, get_bin_mask, bin2inf


# RGT
class RGT(nn.Module):
    def __init__(self,
                 up_embed_dim,
                 down_embed_dim,
                 up_vocab_size,
                 down_vocab_size,
                 up_type_num,
                 down_type_num,
                 up_schema_num,
                 up_max_depth,
                 down_max_dist,
                 up_d_model,
                 down_d_model,
                 up_d_ff,
                 down_d_ff,
                 up_head_num,
                 down_head_num,
                 up_layer_num,
                 down_layer_num,
                 hid_size,
                 dropout,
                 up_pad_idx,
                 down_pad_idx,
                 max_oov_num=50,
                 copy=True,
                 rel_share=True,
                 k_v_share=True,
                 mode="concat",
                 cross_atten="AOD+None",
                 up_rel={"DRD", "DBS"},
                 down_rel={"RPR", "LCA"}):
        super(RGT, self).__init__()

        assert up_d_model % up_head_num == 0
        assert down_d_model % down_head_num == 0
        up_d_k = up_d_model // up_head_num
        down_d_k = down_d_model // down_head_num

        self.up_pad_idx = up_pad_idx
        self.down_pad_idx = down_pad_idx

        # embedding layers
        self.up_nodes_embed = nn.Embedding(up_vocab_size, up_embed_dim)
        self.down_nodes_embed = nn.Embedding(down_vocab_size, down_embed_dim)
        self.up_type_embed = nn.Embedding(up_type_num, up_embed_dim)
        self.down_type_embed = nn.Embedding(down_type_num, down_embed_dim)
        self.up_schema_embed = nn.Embedding(up_schema_num, up_d_k)
        self.up_depth_embed = nn.Embedding(2 * up_max_depth + 2, up_d_k)
        self.down_dist_embed = nn.Embedding(2 * down_max_dist + 1, down_d_k)

        # RGT encoder
        self.RGT_encoder = RGTEncoder(up_head_num,
                                      down_head_num,
                                      up_layer_num,
                                      down_layer_num,
                                      up_d_model,
                                      up_d_ff,
                                      down_d_model,
                                      down_d_ff,
                                      dropout,
                                      rel_share=rel_share,
                                      k_v_share=k_v_share,
                                      mode=mode,
                                      cross_atten=cross_atten,
                                      up_rel=up_rel,
                                      down_rel=down_rel)

        self.up_enc_init = nn.Linear(up_embed_dim, up_d_model)
        self.down_enc_init = nn.Linear(down_embed_dim, down_d_model)
        self.drop = nn.Dropout(dropout)

        self.up_layer_norm = nn.LayerNorm(hid_size, eps=1e-6)
        self.down_layer_norm = nn.LayerNorm(hid_size, eps=1e-6)

        # LSTM decoder
        self.LSTM_decoder = LSTMDecoder(down_embed_dim, hid_size,
                                        down_vocab_size, up_d_model,
                                        down_d_model, dropout, max_oov_num,
                                        copy)
        self.up_dec_prj = nn.Linear(up_d_model, hid_size)
        self.down_dec_prj = nn.Linear(down_d_model, hid_size)

    def encode(self,
               up_x,
               up_type_x,
               down_x,
               down_type_x,
               up_depth,
               up_schema,
               down_dist,
               down_lca,
               AOA_mask=None,
               AOD_mask=None):
        # embed
        up_nodes = self.up_nodes_embed(up_x) + self.up_type_embed(up_type_x)
        down_nodes = self.down_nodes_embed(down_x) + self.down_type_embed(
            down_type_x)
        up_depth = self.up_depth_embed(up_depth)
        up_schema = self.up_schema_embed(up_schema)
        down_dist = self.down_dist_embed(down_dist)

        up_mask = get_bin_mask(up_x, self.up_pad_idx)
        down_mask = get_bin_mask(down_x, self.down_pad_idx)

        up_nodes = self.drop(self.up_enc_init(up_nodes))
        down_nodes = self.drop(self.down_enc_init(down_nodes))

        up_nodes, down_nodes = self.RGT_encoder(up_nodes, down_nodes, up_depth,
                                                up_schema, down_dist, down_lca,
                                                AOA_mask, AOD_mask, up_mask,
                                                down_mask)

        up_nodes = self.drop(self.up_dec_prj(up_nodes))
        down_nodes = self.drop(self.down_dec_prj(down_nodes))

        up_nodes = self.up_layer_norm(up_nodes)
        down_nodes = self.down_layer_norm(down_nodes)

        # [bsz, 1, hid_size]
        h = self.get_init_hidden(up_nodes, down_nodes, up_mask.unsqueeze(-1),
                                 down_mask.unsqueeze(-1))
        h = h.transpose(0, 1)
        c = torch.zeros_like(h)
        hidden = (h, c)

        return up_nodes, down_nodes, hidden, up_mask, down_mask

    @staticmethod
    def get_init_hidden(up_nodes, down_nodes, up_mask=None, down_mask=None):
        # mean pooling
        up_hidden = max_pooling(up_nodes, up_mask)
        down_hidden = max_pooling(down_nodes, down_mask)

        return up_hidden + down_hidden

    def decode(self, questions, up_nodes, down_nodes, hidden, up_mask,
               down_mask, copy_mask, src2trg_map):
        out, hidden = self.LSTM_decoder(questions, hidden, up_nodes,
                                        down_nodes,
                                        bin2inf(up_mask.unsqueeze(1)),
                                        bin2inf(down_mask.unsqueeze(1)),
                                        bin2inf(copy_mask.unsqueeze(1)),
                                        src2trg_map)
        return out, hidden

    def forward(self,
                up_x,
                up_type_x,
                down_x,
                down_type_x,
                up_depth,
                up_schema,
                down_dist,
                down_lca,
                q_x,
                AOA_mask=None,
                AOD_mask=None,
                copy_mask=None,
                src2trg_map=None):

        up_nodes, down_nodes, hidden, up_mask, down_mask = self.encode(
            up_x, up_type_x, down_x, down_type_x, up_depth, up_schema,
            down_dist, down_lca, AOA_mask, AOD_mask)

        questions = self.down_nodes_embed(q_x)

        out, hidden = self.decode(questions, up_nodes, down_nodes, hidden,
                                  up_mask, down_mask, copy_mask, src2trg_map)

        return out


# Transformer base
class TransformerBase(nn.Module):
    def __init__(self,
                 embed_dim,
                 vocab_size,
                 d_model,
                 d_ff,
                 head_num,
                 layer_num,
                 hid_size,
                 dropout,
                 max_oov_num=50,
                 copy=True,
                 rel_share=True,
                 k_v_share=True):
        super(TransformerBase, self).__init__()

        assert d_model % head_num == 0

        self.encoder = RATEncoder(d_model,
                                  d_ff,
                                  head_num,
                                  layer_num,
                                  d_model // head_num,
                                  dropout,
                                  rel_share=rel_share,
                                  k_v_share=k_v_share)

        self.decoder = SimpleLSTMDecoder(embed_dim, hid_size, vocab_size,
                                         max_oov_num, copy)

    def encode(self, nodes, r_k=None, r_v=None, mask=None):
        return self.encoder(nodes, r_k, r_v, mask)

    def decode(self,
               questions,
               nodes,
               hidden,
               mask=None,
               copy_mask=None,
               src2trg_map=None):
        return self.decoder(questions, hidden, nodes, mask, copy_mask,
                            src2trg_map)

    def forward(self,
                nodes,
                questions,
                r_k=None,
                r_v=None,
                mask=None,
                copy_mask=None,
                src2trg_map=None):
        nodes, hidden = self.encode(nodes, r_k, r_v, mask)
        out, _ = self.decode(questions, nodes, hidden, mask, copy_mask,
                             src2trg_map)
        return out


# Normal Transformer
class AbsoluteTransformer(TransformerBase):
    pass


# Transformer with relative position
class RelativeTransformer(TransformerBase):
    def __init__(self,
                 embed_dim,
                 vocab_size,
                 d_model,
                 d_ff,
                 head_num,
                 layer_num,
                 hid_size,
                 dropout,
                 pad_idx,
                 max_dist=4,
                 max_oov_num=50,
                 copy=True,
                 rel_share=True,
                 k_v_share=True):
        assert d_model % head_num == 0
        super(RelativeTransformer, self).__init__(embed_dim,
                                                  vocab_size,
                                                  d_model,
                                                  d_ff,
                                                  head_num,
                                                  layer_num,
                                                  hid_size,
                                                  dropout,
                                                  max_oov_num=max_oov_num,
                                                  copy=copy,
                                                  rel_share=rel_share,
                                                  k_v_share=k_v_share)

        self.d_k = d_model // head_num
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rela_dist_embed = nn.Embedding(2 * max_dist + 1, self.d_k)
        self.pad_idx = pad_idx
        self.k_v_share = k_v_share

        if not self.k_v_share:
            self.r_k_prj = nn.Linear(self.d_k, self.d_k)
            self.r_v_prj = nn.Linear(self.d_k, self.d_k)

        self.init_prj = nn.Linear(embed_dim, d_model)
        self.dec_prj = nn.Linear(d_model, hid_size)
        self.drop = nn.Dropout(dropout)

    def encode(self, nodes, rela_dist):
        mask = self.get_mask(nodes)
        nodes = self.drop(self.init_prj(self.embedding(nodes)))

        r = self.rela_dist_embed(rela_dist)

        if not self.k_v_share:
            r_k = self.r_k_prj(r)
            r_v = self.r_v_prj(r)
        else:
            r_k = r
            r_v = None

        nodes = super(RelativeTransformer,
                      self).encode(nodes, r_k, r_v,
                                   bin2inf(mask).unsqueeze(1).unsqueeze(1))
        nodes = self.drop(self.dec_prj(nodes))

        hidden = self.get_init_hidden(nodes, mask)

        return nodes, hidden, bin2inf(mask).unsqueeze(1)

    def decode(self,
               questions,
               nodes,
               hidden,
               mask=None,
               copy_mask=None,
               src2trg_map=None):
        questions = self.embedding(questions)
        return super(RelativeTransformer,
                     self).decode(questions, nodes, hidden, mask,
                                  bin2inf(copy_mask).unsqueeze(1), src2trg_map)

    @staticmethod
    def get_init_hidden(nodes, mask):
        h = max_pooling(nodes, mask.unsqueeze(-1))
        h = h.transpose(0, 1)
        c = torch.zeros_like(h)
        return (h, c)

    def get_mask(self, x):
        mask = get_bin_mask(x, self.pad_idx)
        return mask

    def forward(self,
                nodes,
                rela_dist,
                questions,
                copy_mask=None,
                src2trg_map=None):
        nodes, hidden, mask = self.encode(nodes, rela_dist)
        out, _ = self.decode(questions, nodes, hidden, mask, copy_mask,
                             src2trg_map)

        return out


# GAT
class GAT(TransformerBase):
    pass


# GCN
class GCN(nn.Module):
    pass


# BiLSTM
class BiLSTM(nn.Module):
    pass


# TreeLSTM
class TreeLSTM(nn.Module):
    pass
