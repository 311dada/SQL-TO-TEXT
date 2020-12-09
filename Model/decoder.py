"""
    ALL DECODER MODULE
        * LSTM Decoder w/o Copy Mechanism
"""
import torch.nn as nn
from Model.attention import BahdanauAttention
import torch
import torch.nn.functional as F


# LSTM Decoder
class LSTMDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_size,
                 output_size,
                 up_d_model,
                 down_d_model,
                 dropout,
                 max_oov_num: int = 50,
                 copy=True):
        super(LSTMDecoder, self).__init__()

        self.lstm = nn.LSTM(hid_size, hid_size, batch_first=True)
        self.up_atten = BahdanauAttention(hid_size, hid_size, hid_size)
        self.down_atten = BahdanauAttention(hid_size, hid_size, hid_size)
        self.copy = copy
        if copy:
            self.copy_score = BahdanauAttention(hid_size,
                                                hid_size,
                                                hid_size,
                                                score=True)
            self.copy_switch = nn.Linear(hid_size, 1)
        self.max_oov_num = max_oov_num
        self.output_size = output_size
        self.out = nn.Linear(hid_size, output_size)
        self.input_prj = nn.Linear(input_dim + hid_size * 2, hid_size)

    def forward(self,
                questions,
                hidden,
                up_nodes,
                down_nodes,
                up_mask=None,
                down_mask=None,
                copy_mask=None,
                src2trg_map=None):

        dec_outputs = []

        if self.copy:
            scatter_matrix = F.one_hot(src2trg_map,
                                       num_classes=self.output_size +
                                       self.max_oov_num).type(torch.float)
        else:
            scatter_matrix = None

        for step in range(questions.size(1)):
            hid = hidden[0].transpose(0, 1)
            up_atten_vec = self.up_atten(hid, up_nodes, up_mask)
            down_atten_vec = self.down_atten(hid, down_nodes, down_mask)
            q_input_vec = questions[:, step].unsqueeze(1)

            cur_input = torch.cat([q_input_vec, up_atten_vec, down_atten_vec],
                                  dim=-1)
            cur_input = self.input_prj(cur_input)
            cur_output, hidden = self.lstm(cur_input, hidden)

            out_score = torch.softmax(self.out(cur_output), dim=-1)

            # copy mechanism
            if self.copy:
                padded = torch.zeros((out_score.size(0), 1, self.max_oov_num),
                                     device=out_score.device)
                gen_score = torch.cat([out_score, padded], dim=-1)
                _, down_copy_score = self.copy_score(hid, down_nodes,
                                                     copy_mask)
                copy_score = down_copy_score.matmul(scatter_matrix)
                p_gen = torch.sigmoid(self.copy_switch(hid))
                out_score = p_gen * gen_score + (1 - p_gen) * copy_score

            dec_outputs.append(out_score)

        return torch.cat(dec_outputs, dim=1), hidden


# Simple LSTM Decoder
class SimpleLSTMDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_size,
                 output_size,
                 max_oov_num: int = 50,
                 copy=True):
        super(SimpleLSTMDecoder, self).__init__()

        self.lstm = nn.LSTM(hid_size, hid_size, batch_first=True)
        self.atten = BahdanauAttention(hid_size, hid_size, hid_size)
        self.copy = copy
        if copy:
            self.copy_score = BahdanauAttention(hid_size,
                                                hid_size,
                                                hid_size,
                                                score=True)
            self.copy_switch = nn.Linear(hid_size, 1)
        self.max_oov_num = max_oov_num
        self.output_size = output_size
        self.out = nn.Linear(hid_size, output_size)
        self.input_prj = nn.Linear(input_dim + hid_size, hid_size)

    def forward(self,
                questions,
                hidden,
                nodes,
                mask=None,
                copy_mask=None,
                src2trg_map=None):

        dec_outputs = []

        if self.copy:
            scatter_matrix = F.one_hot(src2trg_map,
                                       num_classes=self.output_size +
                                       self.max_oov_num).type(torch.float)
        else:
            scatter_matrix = None

        for step in range(questions.size(1)):
            hid = hidden[0].transpose(0, 1)
            atten_vec = self.atten(hid, nodes, mask)
            q_input_vec = questions[:, step].unsqueeze(1)

            cur_input = torch.cat([q_input_vec, atten_vec], dim=-1)
            cur_input = self.input_prj(cur_input)
            cur_output, hidden = self.lstm(cur_input, hidden)

            out_score = torch.softmax(self.out(cur_output), dim=-1)

            # copy mechanism
            if self.copy:
                padded = torch.zeros((out_score.size(0), 1, self.max_oov_num),
                                     device=out_score.device)
                gen_score = torch.cat([out_score, padded], dim=-1)
                _, down_copy_score = self.copy_score(hid, nodes, copy_mask)
                copy_score = down_copy_score.matmul(scatter_matrix)
                p_gen = torch.sigmoid(self.copy_switch(hid))
                out_score = p_gen * gen_score + (1 - p_gen) * copy_score

            dec_outputs.append(out_score)

        return torch.cat(dec_outputs, dim=1), hidden
