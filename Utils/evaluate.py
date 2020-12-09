'''
@Author: your name
@Date: 2020-07-28 17:51:29
LastEditTime: 2020-08-22 14:58:39
LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /Graph2Seq_v2/Utils/evaluate.py
'''
from Utils.search import greedy_search
from Utils.metric import get_metric
from Data.wikisql.dataset import WikisqlDataset
import torch.nn as nn
from Data.vocab import Vocabulary
import torch


def evaluate(data_set: WikisqlDataset,
             model: nn.Module,
             vocab: Vocabulary,
             device: torch.device,
             Model: str,
             delexicalize: bool = False,
             out_file: str = None,
             max_len: int = None,
             batch_size: int = 32) -> float:
    """Evaluate for data_set and return the metric.

    Args:
        data_set (WikisqlDataset): dataset to evaluate.
        model (nn.Module): model to evaluate.
        vocab (Vocabulary): vocabulary.
        device (torch.device): device
        delexicalize (bool, optional): whether to delexicalize. Defaults to False.
        out_file (str, optional): output file. Defaults to None.
        max_len (int, optional): maximum length to decode. Defaults to None.

    Returns:
        float: metric value.
    """

    total_preds = greedy_search(data_set, model, device,
                                vocab.get_idx("<pad>"), Model, max_len,
                                batch_size)

    bleu = get_metric(total_preds, data_set.origin_questions, vocab,
                      delexicalize, data_set.val_map_list, out_file,
                      data_set.idx2tok_map_list)

    return bleu
