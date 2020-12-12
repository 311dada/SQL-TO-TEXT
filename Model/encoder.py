"""
    ALL ENCODER MODULE
        * RGT
        * Transformer
        * GCN
        * BiLSTM
        * TreeLSTM
"""
import torch.nn as nn
from Model.layer import RATLayer, GCNLayer
from Model.attention import BahdanauAttention
import torch
from Model.utils import bin2inf
import torch.nn.functional as F


# RGT encoder
class RGTEncoder(nn.Module):
    def __init__(self,
                 up_head_num,
                 down_head_num,
                 up_layer_num,
                 down_layer_num,
                 up_d_model,
                 up_d_ff,
                 down_d_model,
                 down_d_ff,
                 dropout,
                 rel_share=True,
                 k_v_share=True,
                 mode="concat",
                 cross_atten="AOD+None",
                 up_rel={"DRD", "DBS"},
                 down_rel={"RPR", "LCA"}):
        super(RGTEncoder, self).__init__()

        assert up_rel.issubset({"DRD", "DBS"})
        assert down_rel.issubset({"RPR", "LCA"})
        assert up_layer_num % down_layer_num == 0 or down_layer_num % up_layer_num == 0

        self.up_layer_num = up_layer_num
        self.up_d_k = up_d_model // up_head_num
        self.down_layer_num = down_layer_num
        self.down_d_k = down_d_model // down_head_num

        if mode == "concat":
            self.up_d_rel = self.up_d_k
            self.down_d_rel = self.down_d_k
        elif mode == "multi-head":
            self.up_d_rel = self.up_d_k * len(up_rel)
            self.down_d_rel = self.down_d_k * len(down_rel)
        self.up_to_down_atten, self.down_to_up_atten = cross_atten.split("+")
        self.up_rel = up_rel
        self.down_rel = down_rel
        self.rel_share = rel_share
        self.k_v_share = k_v_share
        self.mode = mode

        assert self.up_to_down_atten in {'None', 'AOD', 'AOF'}
        assert self.down_to_up_atten in {'None', 'AOF', 'AOA'}

        self.up_RAT_layers = nn.ModuleList([
            RATLayer(up_head_num, up_d_model, up_d_ff, self.up_d_rel, dropout,
                     rel_share, k_v_share, mode)
            for _ in range(self.up_layer_num)
        ])

        self.down_RAT_layers = nn.ModuleList([
            RATLayer(down_head_num, down_d_model, down_d_ff, self.down_d_rel,
                     dropout, rel_share, k_v_share, mode)
            for _ in range(self.down_layer_num)
        ])

        self.up_step = self.up_layer_num // self.down_layer_num
        self.down_step = self.down_layer_num // self.up_layer_num

        if not self.up_step:
            self.up_step = 1
        if not self.down_step:
            self.down_step = 1

        if self.up_to_down_atten != 'None':
            self.up_to_down_atten_layers = nn.ModuleList([
                BahdanauAttention(up_d_model, down_d_model, up_d_model)
                for _ in range(self.up_layer_num // self.up_step)
            ])
        if self.down_to_up_atten != 'None':
            self.down_to_up_atten_layers = nn.ModuleList([
                BahdanauAttention(down_d_model, up_d_model, down_d_model)
                for _ in range(self.down_layer_num // self.down_step)
            ])

        self.down_prj = nn.Linear(down_d_model, up_d_model)
        self.up_prj = nn.Linear(up_d_model, down_d_model)

        self.up_layer_norm = nn.LayerNorm(up_d_model, eps=1e-6)
        self.down_layer_norm = nn.LayerNorm(down_d_model, eps=1e-6)

        if "LCA" in self.down_rel:
            self.lca_prj = nn.Linear(up_d_model, self.down_d_k)

        if mode == "concat":
            self.up_r_prj = nn.Linear(self.up_d_k * len(self.up_rel),
                                      self.up_d_k)
            self.down_r_prj = nn.Linear(self.down_d_k * len(self.down_rel),
                                        self.down_d_k)
            if not self.k_v_share:
                self.up_r_k_prj = nn.Linear(self.up_d_k, self.up_d_k)
                self.up_r_v_prj = nn.Linear(self.up_d_k, self.up_d_k)

                self.down_r_k_prj = nn.Linear(self.down_d_k, self.down_d_k)
                self.down_r_v_prj = nn.Linear(self.down_d_k, self.down_d_k)
        elif mode == "multi-head":
            if not self.k_v_share:
                up_dim = self.up_d_k * len(self.up_rel)
                down_dim = self.down_d_k * len(self.down_rel)
                self.up_r_k_prj = nn.Linear(up_dim, up_dim)
                self.up_r_v_prj = nn.Linear(up_dim, up_dim)

                self.down_r_k_prj = nn.Linear(down_dim, down_dim)
                self.down_r_v_prj = nn.Linear(down_dim, down_dim)

        self.drop = nn.Dropout(dropout)

    def get_up_rel(self, up_depth, up_schema):
        up_r = None
        if "DRD" in self.up_rel:
            up_r = up_depth
        if "DBS" in self.up_rel:
            if up_r is None:
                up_r = up_schema
            else:
                up_r = torch.cat([up_r, up_schema], -1)

        if up_r is None:
            return None, None

        if self.mode == "concat":
            up_r = self.up_r_prj(up_r)

        elif self.mode == "multi-head":
            pass

        if self.k_v_share:
            up_r_k = up_r
            up_r_v = None
        else:
            up_r_k = self.drop(self.up_r_k_prj(up_r))
            up_r_v = self.drop(self.up_r_v_prj(up_r))

        return up_r_k, up_r_v

    def get_down_rel(self, down_dist, down_lca, up_nodes):
        down_r = None
        if "RPR" in self.down_rel:
            down_r = down_dist

        if "LCA" in self.down_rel:
            down_lca_emb = self.get_lca(down_lca, up_nodes)
            down_lca_emb = self.lca_prj(down_lca_emb)
            if down_r is None:
                down_r = down_lca_emb
            else:
                down_r = torch.cat([down_r, down_lca_emb], -1)

        if down_r is None:
            return None, None

        if self.mode == "concat":
            down_r = self.down_r_prj(down_r)
        elif self.mode == "multi-head":
            pass

        if not self.k_v_share:
            down_r_k = self.drop(self.down_r_k_prj(down_r))
            down_r_v = self.drop(self.down_r_v_prj(down_r))
        else:
            down_r_k = down_r
            down_r_v = None

        return down_r_k, down_r_v

    def forward(self,
                up_nodes,
                down_nodes,
                up_depth,
                up_schema,
                down_dist,
                down_lca,
                AOA_mask=None,
                AOD_mask=None,
                up_mask=None,
                down_mask=None):
        # get r
        # up
        up_r_k, up_r_v = self.get_up_rel(up_depth, up_schema)

        # circle
        cur_up_step, cur_down_step = 0, 0
        for i in range(self.up_layer_num // self.up_step):
            for step in range(self.up_step):
                # upper encoding
                up_nodes = self.up_RAT_layers[cur_up_step + step](
                    up_nodes,
                    r_k=up_r_k,
                    r_v=up_r_v,
                    mask=bin2inf(up_mask).unsqueeze(1).unsqueeze(1))
            cur_up_step += self.up_step

            # down to up attention
            if self.down_to_up_atten != "None":
                down_nodes = self.down_layer_norm(
                    down_nodes + self.down2up_attention(
                        down_nodes, up_nodes, i, AOA_mask, up_mask))

            # get relation
            down_r_k, down_r_v = self.get_down_rel(down_dist, down_lca,
                                                   up_nodes)

            for step in range(self.down_step):
                # down encoding
                down_nodes = self.down_RAT_layers[cur_down_step + step](
                    down_nodes,
                    r_k=down_r_k,
                    r_v=down_r_v,
                    mask=bin2inf(down_mask).unsqueeze(1).unsqueeze(1))
            cur_down_step += self.down_step

            # up to down attention
            if self.up_to_down_atten != "None":
                up_nodes = self.up_layer_norm(
                    up_nodes + self.up2down_attention(up_nodes, down_nodes, i,
                                                      AOD_mask, down_mask))

        return up_nodes, down_nodes

    @staticmethod
    def get_lca(lca, emb):
        # return F.embedding(lca, emb)
        down_node_num = lca.size(1)
        lca = F.one_hot(lca, num_classes=emb.size(1)).reshape(
            emb.size(0), -1, emb.size(1)).type(torch.float)
        lca = lca.matmul(emb)
        lca = lca.reshape(emb.size(0), down_node_num, down_node_num, -1)
        return lca

    def down2up_attention(self,
                          down_nodes,
                          up_nodes,
                          idx,
                          AOA_mask=None,
                          up_mask=None):
        if self.down_to_up_atten == "None":
            return 0
        elif self.down_to_up_atten == "AOA":
            x = self.down_to_up_atten_layers[idx](down_nodes, up_nodes,
                                                  bin2inf(AOA_mask))
        elif self.down_to_up_atten == "AOF":
            pad_mask = bin2inf(up_mask).unsqueeze(1)
            x = self.down_to_up_atten_layers[idx](down_nodes, up_nodes,
                                                  pad_mask)
        else:
            raise NotImplementedError(
                "Not supported attention from down to up.")

        return self.up_prj(x)

    def up2down_attention(self,
                          up_nodes,
                          down_nodes,
                          idx,
                          AOD_mask=None,
                          down_mask=None):
        if self.up_to_down_atten == "None":
            return 0
        elif self.up_to_down_atten == "AOD":
            x = self.up_to_down_atten_layers[idx](up_nodes, down_nodes,
                                                  bin2inf(AOD_mask))
        elif self.up_to_down_atten == "AOF":
            pad_mask = bin2inf(down_mask).unsqueeze(1)
            x = self.up_to_down_atten_layers[idx](up_nodes, down_nodes,
                                                  pad_mask)
        else:
            raise NotImplementedError(
                "Not supported attention from up to down.")

        return self.down_prj(x)


# RAT encoder
class RATEncoder(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff,
                 head_num,
                 layer_num,
                 d_rel,
                 dropout,
                 rel_share=True,
                 k_v_share=True):
        super(RATEncoder, self).__init__()

        self.layer_num = layer_num

        self.layers = nn.ModuleList([
            RATLayer(head_num,
                     d_model,
                     d_ff,
                     d_rel,
                     dropout,
                     rel_share=rel_share,
                     k_v_share=k_v_share) for _ in range(layer_num)
        ])

    def forward(self, nodes, r_k=None, r_v=None, mask=None):
        for i in range(self.layer_num):
            nodes = self.layers[i](nodes, r_k, r_v, mask)
        return nodes


# GCN Encoder
class GCNEncoder(nn.Module):
    def __init__(self, hid_size, layer_num, dropout) -> None:
        super(GCNEncoder, self).__init__()

        self.layer_num = layer_num
        self.layers = nn.ModuleList(
            [GCNLayer(hid_size, dropout) for _ in range(self.layer_num)])

    def forward(self, x, graph):
        for i in range(self.layer_num):
            x = self.layers[i](x, graph)

        return x
