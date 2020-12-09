'''
Author: your name
Date: 1970-01-01 01:00:00
LastEditTime: 2020-09-08 03:59:35
LastEditors: Please set LastEditors
Description: search
FilePath: /Tree2Seq/Utils/search.py
'''
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from typing import List
from Data.spider.data_utils import get_RAT_batch_data
from Utils.tools import get_position_relations


def greedy_search(
    data_set: Dataset,
    model: nn.Module,
    device: torch.device,
    pad_index: int,
    Model: str,
    max_len: int = None,
    batch_size: int = 32,
) -> List[List]:
    """Greedy search.

    Args:
        data_set (Dataset): data set.
        model (nn.Module): model.
        device (torch.device): device.
        pad_index (int): pad index.
        max_len (int, optional): maximum length to decode. Defaults to None.
        model_format (str, optional): . Defaults to None.

    Returns:
        List[List]: [description]
    """
    options = Model.split("+")
    MODEL = options[0]
    UNK_IDX = data_set.vocab.stoi["<unk>"]
    MAX_RELATIVE_DIS = 4
    PAD_INDEX = pad_index

    total_preds = []
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)

    down_relative_pos_total = torch.tensor(get_position_relations(
        150, MAX_RELATIVE_DIS),
                                           dtype=torch.long,
                                           device=device)
    for batch_data in data_loader:
        if MODEL == "RAT":
            down_nodes, up_nodes, up_nodes_types, down_nodes_types, up_graphs, up_depths, up_schemas, down_to_up_relations, down_relative_pos, questions, copy_masks, src2trg_mapping, mixed_head_masks, up_to_down_masks, down_to_up_masks = get_RAT_batch_data(
                batch_data,
                down_relative_pos_total,
                PAD_INDEX,
                device,
                MAX_RELATIVE_DIS,
                data=data_set.data)

        max_dec_length = max_len or questions.size(-1) - 1

        if MODEL in ["RAT", "SAT", "RAT-1"]:

            inputs = questions[:, 0].reshape(-1, 1)

            nodes_enc, hidden, up_x = model.encoder(down_nodes,
                                                    up_nodes,
                                                    up_nodes_types,
                                                    down_nodes_types,
                                                    up_graphs,
                                                    up_depths,
                                                    up_schemas,
                                                    down_to_up_relations,
                                                    down_relative_pos,
                                                    None,
                                                    up_to_down_masks,
                                                    down_to_up_masks,
                                                    mode="test")

            src_mask = model._get_mask(down_nodes).unsqueeze(1)
            up_mask = model._get_mask(up_nodes).unsqueeze(1)

            preds = []
            for step in range(max_dec_length):
                output, hidden, _, _ = model.decoder(
                    inputs,
                    hidden,
                    nodes_enc,
                    src_mask=src_mask,
                    up_nodes=up_x,
                    up_mask=up_mask,
                    copy_masks=copy_masks,
                    src2trg_mapping=src2trg_mapping,
                    mode="test")
                pred = output.argmax(dim=-1)
                preds.append(pred)
                pred[pred >= len(data_set.vocab)] = UNK_IDX
                inputs = pred

            preds = torch.cat(preds, dim=1)

            total_preds += preds.tolist()

    return total_preds
