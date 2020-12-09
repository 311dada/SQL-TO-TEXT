'''
Author: your name
Date: 1970-01-01 01:00:00
LastEditTime: 2020-09-07 16:04:46
LastEditors: Please set LastEditors
Description: wikisql dataset
FilePath: /Tree2Seq/Data/wikisql/dataset.py
'''
import torch
from torch.utils.data import Dataset
from Data.vocab import Vocabulary
from Data.wikisql.data_utils import load_data, load_seq2seq_data
from typing import Tuple


class WikisqlDataset(Dataset):
    def __init__(self,
                 data_file: str,
                 table_file: str,
                 vocab: Vocabulary = None,
                 min_freq=1,
                 max_depth=4):
        """
        Args:
            data_file (str): data file path.
            table_file (str): table file path.
            vocab (Vocabulary, optional): vocabulary. If none, build it from the data. Defaults to None.
            min_freq (int, optional): the minimum frequency of words in vocabulary. Defaults to 1.
        """
        super(WikisqlDataset, self).__init__()

        down_nodes, up_nodes, up_nodes_types, down_nodes_types, up_graphs, up_depths, down_to_up_relations, questions, copy_masks, src2trg_map_list, mixed_head_masks, origin_ques, vocab, val_map_list, idx2tok_map_list, up_to_down_masks, down_to_up_masks = load_data(
            data_file, table_file, vocab, min_freq, max_depth)

        self.down_nodes = torch.tensor(down_nodes, dtype=torch.long)
        self.up_nodes = torch.tensor(up_nodes, dtype=torch.long)
        self.up_nodes_types = torch.tensor(up_nodes_types, dtype=torch.long)
        self.down_nodes_types = torch.tensor(down_nodes_types,
                                             dtype=torch.long)
        self.up_graphs = torch.tensor(up_graphs, dtype=torch.long)
        self.up_depths = torch.tensor(up_depths, dtype=torch.long)
        self.down_to_up_relations = torch.tensor(down_to_up_relations,
                                                 dtype=torch.long)
        self.questions = torch.tensor(questions, dtype=torch.long)
        self.copy_masks = torch.tensor(copy_masks, dtype=torch.long)
        self.src2trg_mapping = torch.tensor(src2trg_map_list, dtype=torch.long)
        self.mixed_head_masks = torch.tensor(mixed_head_masks,
                                             dtype=torch.long)

        self.up_to_down_masks = torch.tensor(up_to_down_masks,
                                             dtype=torch.long)
        self.down_to_up_masks = torch.tensor(down_to_up_masks,
                                             dtype=torch.long)

        self.origin_questions = origin_ques
        self.vocab = vocab
        self.val_map_list = val_map_list
        self.idx2tok_map_list = idx2tok_map_list
        self.data = "wikisql"

    def __len__(self):
        return self.down_nodes.size(0)

    def __getitem__(self, idx: int) -> Tuple:
        return self.down_nodes[idx], self.up_nodes[idx], self.up_nodes_types[
            idx], self.down_nodes_types[idx], self.up_graphs[
                idx], self.up_depths[idx], self.down_to_up_relations[
                    idx], self.questions[idx], self.copy_masks[
                        idx], self.src2trg_mapping[idx], self.mixed_head_masks[
                            idx], self.up_to_down_masks[
                                idx], self.down_to_up_masks[idx]


class WikisqlSeq2SeqDataset(Dataset):
    def __init__(self,
                 data_file: str,
                 table_file: str,
                 vocab: Vocabulary = None,
                 min_freq=1):
        super(WikisqlSeq2SeqDataset, self).__init__()
        # load data
        sqls, questions, cliques, copy_masks, origin_ques, vocab, val_map_list, clique_graphs, src2trg_map_list, idx2tok_map_list = load_seq2seq_data(
            data_file, table_file, vocab, min_freq)

        self.sqls = torch.tensor(sqls, dtype=torch.long)
        self.questions = torch.tensor(questions, dtype=torch.long)
        self.cliques = torch.tensor(cliques, dtype=torch.long)
        self.copy_masks = torch.tensor(copy_masks, dtype=torch.long)
        self.clique_graphs = torch.tensor(clique_graphs, dtype=torch.long)
        self.src2trg_map_list = torch.tensor(src2trg_map_list,
                                             dtype=torch.long)
        self.origin_questions = origin_ques
        self.vocab = vocab
        self.val_map_list = val_map_list
        self.idx2tok_map_list = idx2tok_map_list

    def __getitem__(self, idx):
        return self.sqls[idx], self.questions[idx], self.cliques[
            idx], self.copy_masks[idx], self.clique_graphs[
                idx], self.src2trg_map_list[idx]

    def __len__(self):
        return self.sqls.size(0)
