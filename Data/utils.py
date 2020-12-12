'''
Author: your name
Date: 2020-08-18 19:58:06
LastEditTime: 2020-09-07 16:11:11
LastEditors: Please set LastEditors
Description: some utilities used by both wikisql and spider
FilePath: /hpc/Data/utils.py
'''
from typing import List, Tuple, Dict
from Utils.tree import flatten, TreeNode, get_all_nodes, set_max_depth, linearize
from Data.vocab import Vocabulary
from collections import Counter
import numpy as np
from Utils.const import SCHEMA_RELATIONS, TYPES, UP_TYPES, DOWN_TYPES
import torch
from Utils.tools import get_position_relations


def rm_contents_in_brackets(contents: str, bracket: str = "()") -> str:
    """Remove somethind in brackets.

    Args:
        contents (str): content

    Returns:
        str: normalized string.
    """
    new_contents = []
    if bracket[0] in contents:
        index = 0
        length = len(contents)
        pre = 0

        while index < length:
            if contents[index] == bracket[0]:
                pre = 1
            elif contents[index] == bracket[1]:
                pre = 0
            elif not pre:
                new_contents.append(contents[index])
            index += 1

    else:
        new_contents = contents
    return ''.join(new_contents)


def get_flatten_data(
        trees: List[TreeNode],
        tables_map=None,
        dbs=None,
        max_depth=4,
        use_schema=True) -> Tuple[List[List], List[List], List[List]]:
    """Get flattened data from tree list.

    Args:
        trees (List[TreeNode]): tree list.

    Returns:
        Tuple[List[List], List[List], List[List], List[List]]: sqls (token), copy masks, cliques
    """
    _ = set_max_depth(trees)
    down_nodes, up_nodes, up_nodes_types, down_nodes_types, up_graphs, up_depths, up_schemas, copy_masks, mixed_head_masks, up_to_down_masks, down_to_up_masks = [], [], [], [], [], [], [], [], [], [], []
    leaf_num = 0
    for index, tree in enumerate(trees):
        seq_data, graph_data, graph, depth, head_mask, up_to_down_mask, down_to_up_mask = flatten(
            tree, max_depth)
        down_node = list(map(lambda node: node.name, seq_data))
        copy_mask = list(map(lambda node: node.copy_mark, seq_data))
        down_nodes_type = list(
            map(lambda node: DOWN_TYPES[node.type], seq_data))
        up_nodes_type = list(map(lambda node: UP_TYPES[node.type], graph_data))
        down_nodes.append(down_node)
        copy_masks.append(copy_mask)
        down_nodes_types.append(down_nodes_type)
        up_nodes_types.append(up_nodes_type)

        # add graph data
        up_node = list(map(lambda node: node.name, graph_data))
        up_graphs.append(graph)
        up_depths.append(depth)
        up_nodes.append(up_node)
        mixed_head_masks.append(head_mask)
        down_to_up_masks.append(down_to_up_mask)

        leaf_num = max(leaf_num, len(down_node))

        if use_schema:
            db_id = dbs[index]
            up_schemas.append(get_up_schema(graph_data, tables_map[db_id]))
        up_to_down_masks.append(up_to_down_mask)

    mixed_head_masks = pad_head_mask(mixed_head_masks, leaf_num)

    return down_nodes, up_nodes, up_nodes_types, down_nodes_types, up_graphs, up_depths, up_schemas, copy_masks, mixed_head_masks, up_to_down_masks, down_to_up_masks


def get_single_graph_data(trees):
    nodes, types, graphs, copy_masks = [], [], [], []
    for tree in trees:
        node_, graph_, _ = linearize(tree, 0)
        nodes_ = list(map(lambda x: x.name, node_))
        types_ = list(map(lambda x: TYPES[x.type], node_))
        copy_mask = list(map(lambda x: x.copy_mark, node_))
        nodes.append(nodes_)
        types.append(types_)
        graphs.append(graph_)
        copy_masks.append(copy_mask)

    return nodes, types, graphs, copy_masks


def build_graph(graphs, node_num):
    Graphs = []

    idx = range(node_num)
    for graph in graphs:
        G = np.zeros((node_num, node_num))
        start, end = list(zip(*graph))
        G[start, end] = 1
        # diag
        G[idx, idx] = 1
        Graphs.append(np.expand_dims(G, 0))

    return np.concatenate(Graphs, axis=0)


def get_up_schema(graph_data, table_map):
    ans = []
    length = len(graph_data)

    foreign_keys = table_map["foreign"]
    primary_keys = table_map["primary"]

    foreign_table_pairs, foreign_column_pairs, foreign_table_column, foreign_column_table = set(
    ), set(), set(), set()
    primary_table_column, primary_column_table = set(), set()

    for foreign_key in foreign_keys:
        table1, column1 = foreign_key[0].split(".")
        table2, column2 = foreign_key[1].split(".")
        foreign_table_pairs.add((table1, table2))
        foreign_table_column.add((table1, column2))
        foreign_column_table.add((column1, table2))
        foreign_column_pairs.add((column1, column2))

    for primary_key in primary_keys:
        table, column = primary_key.split(".")
        primary_table_column.add((table, column))
        primary_column_table.add((column, table))

    for i in range(length):
        for j in range(i + 1, length):
            start, end = graph_data[i], graph_data[j]

            if start.schema is not None and end.schema is not None and start.schema != "*" and end.schema != "*":
                if "." not in start.schema:
                    start_table = start.schema
                    start_column = None
                else:
                    start_table, start_column = start.schema.split(".")

                if "." not in end.schema:
                    end_table = end.schema
                    end_column = None
                else:
                    end_table, end_column = end.schema.split(".")

                # Column-Column
                if start_column is not None and end_column is not None:
                    # SAME-TABLE
                    if start_table == end_table:
                        ans.append([i, j, SCHEMA_RELATIONS["SAME-TABLE"]])
                        ans.append([j, i, SCHEMA_RELATIONS["SAME-TABLE"]])
                    # FOREIGN-KEY-COL-F
                    elif (start_column, end_column) in foreign_column_pairs:
                        ans.append(
                            [i, j, SCHEMA_RELATIONS["FOREIGN-KEY-COL-F"]])
                        ans.append(
                            [j, i, SCHEMA_RELATIONS["FOREIGN-KEY-COL-R"]])

                    # FOREIGN-KEY-COL-R
                    elif (end_column, start_column) in foreign_column_pairs:
                        ans.append(
                            [i, j, SCHEMA_RELATIONS["FOREIGN-KEY-COL-R"]])
                        ans.append(
                            [j, i, SCHEMA_RELATIONS["FOREIGN-KEY-COL-F"]])

                    else:
                        ans.append([i, j, SCHEMA_RELATIONS["NONE"]])
                        ans.append([j, i, SCHEMA_RELATIONS["NONE"]])

                # Column-Table
                elif start_column is not None and end_column is None:
                    # PRIMARY-KEY-F
                    if start_table == end_table and (
                            start_column, end_table) in primary_column_table:
                        ans.append([i, j, SCHEMA_RELATIONS["PRIMARY-KEY-F"]])
                        ans.append([j, i, SCHEMA_RELATIONS["PRIMARY-KEY-R"]])
                    # BELONGS-TO-F
                    elif start_table == end_table:
                        ans.append([i, j, SCHEMA_RELATIONS["BELONGS-TO-F"]])
                        ans.append([j, i, SCHEMA_RELATIONS["BELONGS-TO-R"]])

                    else:
                        ans.append([i, j, SCHEMA_RELATIONS["NONE"]])
                        ans.append([j, i, SCHEMA_RELATIONS["NONE"]])

                # Table-Column
                elif end_column is not None and start_column is None:
                    # PRIMARY-KEY-R
                    if start_table == end_table and (
                            start_table, end_column) in primary_table_column:
                        ans.append([i, j, SCHEMA_RELATIONS["PRIMARY-KEY-R"]])
                        ans.append([j, i, SCHEMA_RELATIONS["PRIMARY-KEY-F"]])

                    # BELONGS-TO-R
                    elif start_table == end_table:
                        ans.append([i, j, SCHEMA_RELATIONS["BELONGS-TO-R"]])
                        ans.append([j, i, SCHEMA_RELATIONS["BELONGS-TO-F"]])

                    else:
                        ans.append([i, j, SCHEMA_RELATIONS["NONE"]])
                        ans.append([j, i, SCHEMA_RELATIONS["NONE"]])

                # Table-Table
                else:
                    # FOREIGN-KEY-TAB-B
                    if (start_table, end_table) in foreign_table_pairs and (
                            end_table, start_table) in foreign_table_pairs:
                        ans.append(
                            [i, j, SCHEMA_RELATIONS["FOREIGN-KEY-TAB-B"]])
                        ans.append(
                            [j, i, SCHEMA_RELATIONS["FOREIGN-KEY-TAB-B"]])
                    # FOREIGN-KEY-TAB-F
                    elif (start_table, end_table) in foreign_table_pairs:
                        ans.append(
                            [i, j, SCHEMA_RELATIONS["FOREIGN-KEY-TAB-F"]])
                        ans.append(
                            [j, i, SCHEMA_RELATIONS["FOREIGN-KEY-TAB-R"]])
                    # FOREIGN-KEY-TAB-R
                    elif (end_table, start_table) in foreign_table_pairs:
                        ans.append(
                            [i, j, SCHEMA_RELATIONS["FOREIGN-KEY-TAB-R"]])
                        ans.append(
                            [j, i, SCHEMA_RELATIONS["FOREIGN-KEY-TAB-F"]])
                    else:
                        ans.append([i, j, SCHEMA_RELATIONS["NONE"]])
                        ans.append([j, i, SCHEMA_RELATIONS["NONE"]])

    return ans


def pad_head_mask(head_masks, leaf_num):

    paded_head_masks = np.zeros((len(head_masks), 4, leaf_num, leaf_num))
    for index, head_mask in enumerate(head_masks):
        for k in range(4):
            for idx, leaf in enumerate(head_mask[k]):
                paded_head_masks[index][k][idx][leaf] = 1
    return paded_head_masks


def build_vocab(sqls: List[List[str]],
                questions: List[List[str]],
                min_freq: int = 1) -> Vocabulary:
    """Build the vocabulary from sqls and questions.

    Args:
        sqls (List[List[str]]): sqls
        questions (List[List[str]]): questions.
        min_freq (int, optional): minimum frequency. Defaults to 1.

    Returns:
        Vocabulary: vocabulary.
    """
    counter = Counter()

    for sql in sqls:
        for sql_tok in sql:
            counter[sql_tok] += 1

    for ques in questions:
        for ques_tok in ques:
            counter[ques_tok] += 1

    vocab = Vocabulary(counter,
                       min_freq=min_freq,
                       specials=["<pad>", "<unk>", "<bos>", "<eos>"])

    return vocab


def build_up_vocab(up_nodes, min_freq):
    counter = Counter()

    for up_node_list in up_nodes:
        for up_node in up_node_list:
            counter[up_node] += 1
    return Vocabulary(counter, min_freq=min_freq, specials=['<pad>', '<unk>'])


def pad(data: List[List],
        max_len: int = None,
        pad_val: int = None) -> List[List]:
    """Pad pad_val to Data with max_len. If max_len is None, pad to the maximum length
    of data. If pad_val is None, pad 0.

    Args:
        data (List[List]): data.
        max_len (int, optional): maximum length. Defaults to None.
        pad_val (int, optional): value to pad. Defaults to None.

    Returns:
        List[List]: padded data.
    """
    total_len = max_len or max(map(lambda x: len(x), data))
    to_pad = pad_val or 0

    for index, sample in enumerate(data):
        data[index] = sample + [to_pad] * (total_len - len(sample))
    return data


def pad_and_transform_cliques(cliques: List[List],
                              node_num: int) -> np.ndarray:
    """Pad cliques and transform it to the binary format.

    Args:
        cliques (List[List]): cliques.
        node_num (int): node number.

    Returns:
        np.ndarray: (N, clique_num, node_num)
    """
    clique_num = max(map(lambda clique: max(clique), cliques)) + 1
    N = len(cliques)
    new_cliques = np.zeros((N, clique_num, node_num))

    for index, clique in enumerate(cliques):
        clique = np.array(clique + [-1] * (node_num - len(clique)))
        for clique_idx in range(clique.max() + 1):
            new_cliques[index, clique_idx, clique == clique_idx] = 1
    return new_cliques


def get_lcs_list_helper(root: TreeNode,
                        dist: List[List] = None) -> Tuple[List[Tuple], List]:
    """Helprt function for [get_lcs_list] function.

    Args:
        root (TreeNode): tree root.

    Returns:
        Tuple[List[Tuple], List]: lcs_list, nodes_list
    """
    children_lcs_list, children_nodes_list = [], []
    for child in root.children:
        child_lcs_list, child_nodes_list = get_lcs_list_helper(child, dist)
        children_lcs_list += child_lcs_list
        children_nodes_list.append(child_nodes_list)

    lcs_list = children_lcs_list
    nodes_list = [root.idx] + sum(children_nodes_list, [])

    lcs = root.idx
    # lcs_depth = root.depth
    # case 0: root self
    # lcs_list.append((lcs, lcs, lcs_type, dist[lcs][lcs],
    #                  dist[root.idx][root.idx], lcs_depth))
    lcs_list.append((lcs, lcs, root))
    # case 1: root and subtrees
    for child_nodes_list in children_nodes_list:
        # children_root = child_nodes_list[0]
        for node in child_nodes_list:
            # dist[lcs][node] = dist[children_root][node] + 1
            # dist[node][lcs] = dist[children_root][node] + 1
            # lcs_list.append((lcs, node, lcs_type, dist[lcs][lcs],
            #                  dist[node][lcs], lcs_depth))
            # lcs_list.append((node, lcs, lcs_type, dist[node][lcs],
            #                  dist[lcs][lcs], lcs_depth))
            lcs_list.append((lcs, node, root))
            lcs_list.append((node, lcs, root))
    # case 2: subtree and subtree
    subtree_num = len(children_nodes_list)
    for i in range(subtree_num):
        for j in range(subtree_num):
            if i == j:
                continue
            first_subtree = children_nodes_list[i]
            second_subtree = children_nodes_list[j]
            for first in first_subtree:
                for second in second_subtree:
                    # lcs_list.append((first, second, lcs_type, dist[first][lcs],
                    #                  dist[second][lcs], lcs_depth))
                    lcs_list.append((first, second, root))
    return lcs_list, nodes_list


def get_lcs_list(root: TreeNode, max_node_num: int) -> List[Tuple]:
    """Get the lcs lists for any two nodes.

    Args:
        root (TreeNode): tree root.
        max_node_num (int): maximum node number.

    Returns:
        List[Tuple]: a tuple list. Each tuple is
        (node1_idx, node2_idx, lcs_type, lcs_node1_dist, lcs_node2_dist)
    """
    # dist = [[0 for i in range(max_node_num)] for j in range(max_node_num)]
    lcs_list, _ = get_lcs_list_helper(root)
    nodes = get_all_nodes(root)

    new_lcs_list = []
    for tup in lcs_list:
        first = tup[0]
        second = tup[1]

        if not nodes[first].children and not nodes[second].children:
            # new_lcs_list.append((nodes[first].seq_idx, nodes[second].seq_idx,
            #                      tup[2], tup[3], tup[4], tup[5]))
            lcs = tup[2]
            if lcs.graph_idx == -1:
                lcs = lcs.parent
            new_lcs_list.append((nodes[first], nodes[second], lcs))
    return new_lcs_list


def get_lcs_relation(trees: List[TreeNode],
                     max_node_num: int,
                     non_leaf_node_num: int,
                     max_depth: int = 4) -> Tuple[np.ndarray, int]:
    """Get the distance relation.

    Args:
        trees (List[TreeNode]): tree list.
        max_node_num (int): maximum nodes number.
        max_dist (int): the maximum height of all trees.
        NODE_TYPE_NUM (int): nodes type number.

    Returns:
        Tuple[np.ndarray, int]: (N, 3, max_node_num, max_node_num), relation_num
    """

    # return lcs_relation, relation_num
    lcs_relations = []
    # relation_masks = []
    # max_relation = 0

    for index, tree in enumerate(trees):
        lcs_relation = [[0 for i in range(max_node_num)]
                        for j in range(max_node_num)]

        # relation_mask = [[[] for j in range(max_node_num)]
        #                  for k in range(max_node_num)]
        lcs_list = get_lcs_list(tree, max_node_num)

        for lcs_ in lcs_list:
            left, right = lcs_[0], lcs_[1]
            i, j = left.seq_idx, right.seq_idx
            lcs_relation[i][j] = lcs_[2].graph_idx

            # while left.parent is not None and left.parent.graph_idx != lcs_[
            #         2].graph_idx:
            #     left = left.parent
            #     relation_mask[i][j].append(left.graph_idx)

            # while right.parent is not None and right.parent.graph_idx != lcs_[
            #         2].graph_idx:
            #     right = right.parent
            #     relation_mask[i][j].append(right.graph_idx)
            # lcs = lcs_[2]
        #     while lcs is not None:
        #         # relation_mask[i][j][lcs.graph_idx] = 1
        #         relation_mask[i][j].append(lcs.graph_idx)
        #         lcs = lcs.parent
        #     max_relation = max(max_relation, len(relation_mask[i][j]))

        # relation_masks.append(relation_mask)

        lcs_relations.append(lcs_relation)

    # relation_masks = pad_lcs_relation_masks(relation_masks, max_relation)

    return lcs_relations


def pad_lcs_relation_masks_and_depths(relation_masks, relation_depths,
                                      max_len):
    max_node_num = len(relation_masks[0])
    for i in range(len(relation_masks)):
        for j in range(max_node_num):
            for k in range(max_node_num):
                if relation_masks[i][j][k]:
                    relation_masks[i][j][k] += [relation_masks[i][j][k][0]] * (
                        max_len - len(relation_masks[i][j][k]))
                    relation_depths[i][j][k] += [
                        relation_depths[i][j][k][0]
                    ] * (max_len - len(relation_depths[i][j][k]))
                else:
                    relation_masks[i][j][k] += [0] * (
                        max_len - len(relation_masks[i][j][k]))

                    relation_depths[i][j][k] += [0] * (
                        max_len - len(relation_depths[i][j][k]))
    return relation_masks, relation_depths


def pad_lcs_relation_masks(relation_masks, max_len):
    max_node_num = len(relation_masks[0])
    for i in range(len(relation_masks)):
        for j in range(max_node_num):
            for k in range(max_node_num):
                if relation_masks[i][j][k]:
                    relation_masks[i][j][k] += [relation_masks[i][j][k][0]] * (
                        max_len - len(relation_masks[i][j][k]))
                else:
                    relation_masks[i][j][k] += [0] * (
                        max_len - len(relation_masks[i][j][k]))

    return relation_masks


def get_same_column_dict(root: TreeNode, same_column: Dict):
    """Traverse the tree and save the same columns to a dict.

    Args:
        root (TreeNode): the root of a tree.
        same_column (Dict): the dict to save the result.
    """
    if root.col_mark >= 0:
        if root.col_mark not in same_column:
            same_column[root.col_mark] = []
        same_column[root.col_mark].append(root.seq_idx)
    for child in root.children:
        get_same_column_dict(child, same_column)


def get_column_relation(trees: List[TreeNode],
                        max_node_num: int) -> Tuple[np.ndarray, int]:
    """Get the column relation.

    Args:
        trees (List[TreeNode]): tree list.
        max_node_num (int): maximum nodes number.

    Returns:
        Tuple[np.ndarray, int]: (N, 1, max_node_num, max_node_num), relation_num
    """
    N = len(trees)
    col_relation = np.zeros((N, max_node_num, max_node_num))

    for index, tree in enumerate(trees):
        same_column = dict()
        get_same_column_dict(tree, same_column)
        for col in same_column:
            col_idx = same_column[col]
            col_idx_start = np.array(col_idx, dtype=np.int)
            col_idx_end = np.array(col_idx, dtype=np.int).reshape(-1, 1)
            col_relation[index, col_idx_start, col_idx_end] = 1

    col_relation = np.expand_dims(col_relation, axis=1)
    return col_relation, 2


def get_relations(trees: List[TreeNode], max_node_num: int,
                  non_leaf_node_num) -> Tuple[np.ndarray, int]:
    """Get relation representations.

    Args:
        trees (List[TreeNode]): tree list.
        max_node_num (int): maximum nodes number.
        NODE_TYPE_NUM (int): nodes type number.

    Returns:
        Tuple[np.ndarray, int]: (N, relation_num, max_node_num, max_node_num), relation_num
    """
    # lcs relation
    lcs_relation = get_lcs_relation(trees, max_node_num, non_leaf_node_num)
    return lcs_relation


def get_clique_graph(cliques: np.ndarray) -> np.ndarray:
    """Get graph representation.

    Args:
        cliques (np.ndarray): cliques.

    Returns:
        np.ndarray: (N, node_num, node_num)
    """
    N, clique_num, node_num = cliques.shape

    graphs = np.zeros((N, node_num, node_num))
    for i in range(N):
        for j in range(clique_num):
            one_hot = cliques[i][j]
            one_hot_x = np.where(one_hot > 0)[0]
            one_hot_y = one_hot_x.reshape(-1, 1)
            graphs[i, one_hot_x, one_hot_y] = 1
    return graphs


def build_graphs_and_depths(graphs, depths, node_num, max_depth):
    np_graphs = []
    np_depths = []

    for index, position in enumerate(graphs):
        graph = np.zeros((node_num, node_num))
        start, end = list(zip(*position))
        graph[start, end] = 1
        np_graphs.append(graph)

        depth = np.full((node_num, node_num), max_depth * 2 + 1)

        start, end, val = list(zip(*depths[index]))
        depth[start, end] = val
        ids = range(node_num)
        depth[ids, ids] = max_depth
        np_depths.append(depth)

    return np.array(np_graphs), np.array(np_depths)


def get_up_schemas(up_schemas, up_nodes_num):
    np_up_schemas = np.full((len(up_schemas), up_nodes_num, up_nodes_num),
                            SCHEMA_RELATIONS["NONE"])

    for idx, up_schema in enumerate(up_schemas):
        start = list(map(lambda x: x[0], up_schema))
        end = list(map(lambda x: x[1], up_schema))
        relation = list(map(lambda x: x[2], up_schema))

        np_up_schemas[idx, start, end] = relation

    return np_up_schemas


def pad_up_to_down_masks(up_to_down_masks, up_nodes_num, down_nodes_num):
    masks = []
    for sample_idx, up_to_down_mask in enumerate(up_to_down_masks):
        mask = np.zeros((up_nodes_num, down_nodes_num))
        for up_idx, down_mask in enumerate(up_to_down_mask):
            mask[up_idx, down_mask] = 1
        masks.append(mask)

    return np.array(masks)


def get_lens(x, pad):
    return torch.max(torch.sum(x != pad, dim=1))


def get_length(x, pad):
    return torch.sum(x != pad, dim=1)


def get_RGT_batch_data(batch_data, up_pad_idx, down_pad_idx, device, k,
                       down_vocab_size, down_unk_idx):
    down_x, up_x, up_x_type, down_x_type, up_depth, up_schema, down_lca, q_x, copy_mask, src2trg_map, AOD_mask, AOA_mask = batch_data

    up_node_num = get_lens(up_x, up_pad_idx)
    down_node_num = get_lens(down_x, down_pad_idx)
    q_num = get_lens(q_x, down_pad_idx)

    up_x = up_x[:, :up_node_num].to(device)
    down_x = down_x[:, :down_node_num].to(device)
    up_x_type = up_x_type[:, :up_node_num].to(device)
    down_x_type = down_x_type[:, :down_node_num].to(device)
    up_depth = up_depth[:, :up_node_num, :up_node_num].to(device)
    up_schema = up_schema[:, :up_node_num, :up_node_num].to(device)
    down_lca = down_lca[:, :down_node_num, :down_node_num].to(device)
    q_x = q_x[:, :q_num]
    label = q_x[:, 1:].to(device)
    q_x[range(q_x.size(0)), get_length(q_x, down_pad_idx) - 1] = down_pad_idx
    q_x = q_x[:, :-1]
    q_x[q_x >= down_vocab_size] = down_unk_idx
    q_x = q_x.to(device)
    copy_mask = copy_mask[:, :down_node_num].to(device)
    src2trg_map = src2trg_map[:, :down_node_num].to(device)
    AOD_mask = AOD_mask[:, :up_node_num, :down_node_num].to(device)
    AOA_mask = AOA_mask[:, :down_node_num, :up_node_num].to(device)
    down_dist = torch.tensor(get_position_relations(
        down_node_num.item(), k)).expand(up_x.size(0), -1,
                                         -1).type(torch.long).to(device)

    return (up_x, up_x_type, down_x, down_x_type, up_depth, up_schema,
            down_dist, down_lca, q_x, AOA_mask, AOD_mask, copy_mask,
            src2trg_map), label


def get_seq_batch_data(batch_data,
                       pad_idx,
                       device,
                       vocab_size,
                       unk_idx,
                       k=None):
    nodes, q, copy_mask, src2trg_map = batch_data

    node_num = get_lens(nodes, pad_idx).item()
    nodes = nodes[:, :node_num].to(device)
    q_num = get_lens(q, pad_idx).item()
    q_x = q[:, :q_num]
    label = q_x[:, 1:].to(device)
    q_x[range(q_x.size(0)), get_length(q_x, pad_idx) - 1] = pad_idx
    q_x = q_x[:, :-1]
    q_x[q_x >= vocab_size] = unk_idx
    q_x = q_x.to(device)

    copy_mask = copy_mask[:, :node_num].to(device)
    src2trg_map = src2trg_map[:, :node_num].to(device)

    if k is not None:
        rela_dist = torch.tensor(get_position_relations(node_num, k)).expand(
            q.size(0), -1, -1).type(torch.long).to(device)
        return (nodes, q_x, rela_dist, copy_mask, src2trg_map), label
    else:
        return (nodes, q_x, copy_mask, src2trg_map), label


def get_single_graph_batch_data(
    batch_data,
    pad_idx,
    device,
    vocab_size,
    unk_idx,
):
    nodes, types, q, graphs, copy_mask, src2trg_map = batch_data

    node_num = get_lens(nodes, pad_idx).item()
    nodes = nodes[:, :node_num].to(device)
    types = types[:, :node_num].to(device)
    graphs = graphs[:, :node_num, :node_num].to(device)
    q_num = get_lens(q, pad_idx).item()
    q_x = q[:, :q_num]
    label = q_x[:, 1:].to(device)
    q_x[range(q_x.size(0)), get_length(q_x, pad_idx) - 1] = pad_idx
    q_x = q_x[:, :-1]
    q_x[q_x >= vocab_size] = unk_idx
    q_x = q_x.to(device)

    copy_mask = copy_mask[:, :node_num].to(device)
    src2trg_map = src2trg_map[:, :node_num].to(device)

    return (nodes, types, q_x, graphs, copy_mask, src2trg_map), label
