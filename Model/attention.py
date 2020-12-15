"""
    ALL ATTENTION MODULE
        * Multi Head Attention
        * Multi Head Attention with Relation
        * Bahdanau Attention
"""
import torch.nn as nn
import torch
import math

# class BahdanauAttention(nn.Module):
#     def __init__(self, dim1, dim2, dim3, score=False):
#         super(BahdanauAttention, self).__init__()
#         self.W = nn.Linear(dim1, dim3)
#         self.U = nn.Linear(dim2, dim3)
#         self.V = nn.Linear(dim3, 1)
#         self.score = score

#     def forward(self, S, H, pad_mask=None):
#         S_ = self.W(S)
#         H_ = self.U(H)
#         S_H_add = S_.unsqueeze(-2) + H_.unsqueeze(-3)
#         score = self.V(torch.tanh(S_H_add)).squeeze(-1)

#         if pad_mask is not None:
#             score += pad_mask.type(torch.float)
#         score = torch.softmax(score, dim=-1)

#         if not self.score:
#             return score.matmul(H)
#         else:
#             return score.matmul(H), score


class BahdanauAttention(nn.Module):
    def __init__(self, dim1, dim2, dim3, score=False) -> None:
        super(BahdanauAttention, self).__init__()
        self.W = nn.Linear(dim1, dim2)
        self.score = score

    def forward(self, S, H, pad_mask=None):
        S_ = self.W(S)

        score = S_.bmm(H.transpose(-1, -2))
        if pad_mask is not None:
            score += pad_mask.type(torch.float)
        score = torch.softmax(score, dim=-1)

        if not self.score:
            return score.matmul(H)
        else:
            return score.matmul(H), score


class RelationMultiHeadAttention(nn.Module):
    def __init__(self,
                 head_num,
                 d_model,
                 d_rel,
                 rel_share=True,
                 k_v_share=False,
                 mode="concat"):
        super(RelationMultiHeadAttention, self).__init__()

        self.head_num = head_num
        self.d_model = d_model
        self.d_k = self.d_model // self.head_num
        self.mode = mode

        self.rel_share = rel_share
        self.k_v_share = k_v_share

        if mode == "concat":
            self.d_rel = d_rel
        elif mode == "multi-head":
            assert d_rel % self.head_num == 0
            self.d_rel = self.d_k
        else:
            raise NotImplementedError("Not supported RAT mode.")

        if not self.rel_share:
            self.r_k_prj = nn.Linear(self.d_rel, self.d_k)
            if not self.k_v_share:
                self.r_v_prj = nn.Linear(self.d_rel, self.d_k)

        # prj layers
        self.prj_q = nn.Linear(d_model, d_model)
        self.prj_k = nn.Linear(d_model, d_model)
        self.prj_v = nn.Linear(d_model, d_model)
        self.final_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, r_k=None, r_v=None, mask=None):
        assert (not self.k_v_share and r_v is not None) or r_v is None
        node_num = q.size(1)

        if r_k is not None:
            # calculate r_k and r_v
            # r_k first
            if self.mode == "concat":
                if not self.rel_share:
                    r_k = self.r_k_prj(r_k)
            elif self.mode == "multi-head":
                r_num = r_k.size(-1) // self.d_k
                r_k = r_k.reshape(-1, node_num, node_num,
                                  r_num, self.d_rel).unsqueeze(1).transpose(
                                      1, 4).squeeze(4)
                r_k = r_k.repeat(1, self.head_num // r_num, 1, 1, 1)
                if not self.rel_share:
                    r_k = self.r_k_prj(r_k)

            # r_v then
            if self.k_v_share:
                r_v = r_k
            else:
                if self.mode == "concat":
                    if not self.rel_share:
                        r_v = self.r_v_prj(r_v)
                elif self.mode == "multi-head":
                    r_num = r_v.size(-1) // self.d_k
                    r_v = r_v.reshape(-1, node_num, node_num, r_num,
                                      self.d_rel).unsqueeze(1).transpose(
                                          1, 4).squeeze(4)
                    r_v = r_v.repeat(1, self.head_num // r_num, 1, 1, 1)
                    if not self.rel_share:
                        r_v = self.r_v_prj(r_v)

        q = self.prj_q(q)
        k = self.prj_k(k)
        v = self.prj_v(v)

        bsz = q.size(0)

        def shape(x):
            return x.view(bsz, -1, self.head_num, self.d_k).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).reshape(bsz, -1, self.d_model)

        q = shape(q)
        k = shape(k)
        v = shape(v)

        # calculate scores: [bsz, head_num, node_num, node_num]
        # part 1
        score_1 = q.matmul(k.transpose(-1, -2)) / math.sqrt(self.d_k)

        if r_k is not None:
            # part 2
            # concat mode
            if self.mode == "concat":
                score_2 = torch.einsum("ijmn,imsn->ijms", q, r_k)
            # multi-head mode
            elif self.mode == "multi-head":
                score_2 = torch.einsum("ijmn,ijmsn->ijms", q, r_k)
            else:
                raise NotImplementedError("Not supported mode.")
        else:
            score_2 = 0
        # [bsz, head_num, node_num, node_num]
        score = score_1 + score_2

        if mask is not None:
            score = score + mask.type(torch.float)
        score = torch.softmax(score, dim=-1)

        # calculate values: [bsz, node_num, d_model]
        # part 1
        value_1 = score.matmul(v)

        if r_k is not None:
            # part 2
            if self.mode == "concat":
                value_2 = torch.einsum("ijmn,imns->ijms", score, r_v)
            # multi-head mode
            elif self.mode == "multi-head":
                value_2 = torch.einsum("ijmn,ijmns->ijms", score, r_v)
            else:
                raise NotImplementedError("Not supported mode.")
        else:
            value_2 = 0

        # [bsz, head_num, node_num, d_k]
        value = value_1 + value_2
        # [bsz, node_num, d_model]
        value = unshape(value)

        return self.final_linear(value)
