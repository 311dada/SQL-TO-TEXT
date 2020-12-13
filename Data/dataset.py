'''
Author: your name
Date: 1970-01-01 01:00:00
LastEditTime: 2020-09-07 15:50:58
LastEditors: Please set LastEditors
Description: wikisql dataset
FilePath: /Tree2Seq/Data/wikisql/dataset.py
'''
from torch.utils.data import Dataset
from Data.vocab import Vocabulary
from Data.spider.data_utils import load_spider_data, load_spider_seq2seq_data, load_spider_single_graph_data, load_spider_tree_data
from typing import Tuple, List
import torch


class RGTDataset(Dataset):
    def __init__(self,
                 data_files: List[str],
                 table_file: str,
                 down_vocab: Vocabulary = None,
                 data="spider",
                 up_vocab=None,
                 min_freq=1,
                 max_depth=4):
        """
        Args:
            data_file (str): data file path.
            table_file (str): table file path.
            vocab (Vocabulary, optional): vocabulary. If none, build it from the data. Defaults to None.
            min_freq (int, optional): the minimum frequency of words in vocabulary. Defaults to 1.
        """
        super(RGTDataset, self).__init__()

        if data == "spider":
            load_data = load_spider_data
        else:
            # TODO
            pass

        down_nodes, up_nodes, up_nodes_types, down_nodes_types, up_depths, up_schemas, down_to_up_relations, questions, copy_masks, src2trg_map_list, origin_ques, down_vocab, up_vocab, val_map_list, idx2tok_map_list, up_to_down_masks, down_to_up_masks = load_data(
            data_files, table_file, down_vocab, up_vocab, min_freq, max_depth)

        self.down_nodes = torch.tensor(down_nodes, dtype=torch.long)
        self.up_nodes = torch.tensor(up_nodes, dtype=torch.long)
        self.up_nodes_types = torch.tensor(up_nodes_types, dtype=torch.long)
        self.down_nodes_types = torch.tensor(down_nodes_types,
                                             dtype=torch.long)
        self.up_depths = torch.tensor(up_depths, dtype=torch.long)
        self.up_schemas = torch.tensor(up_schemas, dtype=torch.long)
        self.down_lca = torch.tensor(down_to_up_relations, dtype=torch.long)
        self.questions = torch.tensor(questions, dtype=torch.long)
        self.copy_masks = torch.tensor(copy_masks, dtype=torch.long)
        self.src2trg_mapping = torch.tensor(src2trg_map_list, dtype=torch.long)

        self.up_to_down_masks = torch.tensor(up_to_down_masks,
                                             dtype=torch.long)

        self.down_to_up_masks = torch.tensor(down_to_up_masks,
                                             dtype=torch.long)

        self.origin_questions = origin_ques
        self.down_vocab = down_vocab
        self.up_vocab = up_vocab
        self.val_map_list = val_map_list
        self.idx2tok_map_list = idx2tok_map_list

    def __len__(self):
        return self.down_nodes.size(0)

    def __getitem__(self, idx: int) -> Tuple:
        return self.down_nodes[idx], self.up_nodes[idx], self.up_nodes_types[
            idx], self.down_nodes_types[idx], self.up_depths[
                idx], self.up_schemas[idx], self.down_lca[idx], self.questions[
                    idx], self.copy_masks[idx], self.src2trg_mapping[
                        idx], self.up_to_down_masks[
                            idx], self.down_to_up_masks[idx]


class SeqDataset(Dataset):
    def __init__(self,
                 data_files,
                 table_file,
                 data="spider",
                 vocab=None,
                 min_freq=1):
        super(SeqDataset, self).__init__()
        if data == "spider":
            load_seq2seq_data = load_spider_seq2seq_data
        else:
            # TODO
            pass
        sqls, questions, copy_masks, origin_ques, vocab, val_map_list, src2trg_map_list, idx2tok_map_list = load_seq2seq_data(
            data_files, table_file, vocab, min_freq)

        self.sqls = torch.tensor(sqls, dtype=torch.long)
        self.questions = torch.tensor(questions, dtype=torch.long)
        self.copy_masks = torch.tensor(copy_masks, dtype=torch.long)
        self.src2trg_mapping = torch.tensor(src2trg_map_list, dtype=torch.long)
        self.origin_questions = origin_ques
        self.vocab = vocab
        self.val_map_list = val_map_list
        self.idx2tok_map_list = idx2tok_map_list

    def __len__(self) -> int:
        return self.sqls.size(0)

    def __getitem__(self, index: int):
        return self.sqls[index], self.questions[index], self.copy_masks[
            index], self.src2trg_mapping[index]


class SingleGraphDataset(Dataset):
    def __init__(self,
                 data_files,
                 table_file,
                 data="spider",
                 vocab=None,
                 min_freq=1):
        super(SingleGraphDataset, self).__init__()
        if data == "spider":
            load_single_graph_data = load_spider_single_graph_data
        else:
            # TODO
            pass
        nodes, types, questions, graphs, copy_masks, origin_ques, vocab, val_map_list, src2trg_map_list, idx2tok_map_list = load_single_graph_data(
            data_files, table_file, vocab, min_freq)

        self.nodes = torch.tensor(nodes, dtype=torch.long)
        self.types = torch.tensor(types, dtype=torch.long)
        self.questions = torch.tensor(questions, dtype=torch.long)
        self.copy_masks = torch.tensor(copy_masks, dtype=torch.long)
        self.src2trg_map = torch.tensor(src2trg_map_list, dtype=torch.long)
        self.graphs = torch.tensor(graphs, dtype=torch.long)

        self.origin_questions = origin_ques
        self.vocab = vocab
        self.val_map_list = val_map_list
        self.idx2tok_map_list = idx2tok_map_list

    def __len__(self) -> int:
        return self.nodes.size(0)

    def __getitem__(self, index: int):
        return self.nodes[index], self.types[index], self.questions[
            index], self.graphs[index], self.copy_masks[
                index], self.src2trg_map[index]


class TreeDataset(Dataset):
    def __init__(self,
                 data_files,
                 table_file,
                 data="spider",
                 vocab=None,
                 min_freq=1):
        super(TreeDataset, self).__init__()

        if data == "spider":
            load_tree_data = load_spider_tree_data
        else:
            # TODO
            load_tree_data = None

        nodes, types, node_order, adjacency_list, edge_order, questions, origin_ques, vocab, val_map_list, copy_mask, src2trg_map, idx2tok_map_list = load_tree_data(
            data_files, table_file, vocab, min_freq)

        self.nodes = nodes
        self.types = types
        self.node_order = node_order
        self.adjacency_list = adjacency_list
        self.edge_order = edge_order
        self.questions = questions
        self.copy_mask = copy_mask
        self.src2trg_map = src2trg_map

        self.origin_questions = origin_ques
        self.vocab = vocab
        self.val_map_list = val_map_list
        self.idx2tok_map_list = idx2tok_map_list

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, index: int):
        return torch.tensor(self.nodes[index], dtype=torch.long), torch.tensor(
            self.types[index], dtype=torch.long), torch.tensor(
                self.node_order[index], dtype=torch.long), torch.tensor(
                    self.adjacency_list[index],
                    dtype=torch.long), torch.tensor(
                        self.edge_order[index],
                        dtype=torch.long), torch.tensor(
                            self.questions[index],
                            dtype=torch.long), torch.tensor(
                                self.copy_mask[index],
                                dtype=torch.long), torch.tensor(
                                    self.src2trg_map[index], dtype=torch.long)
