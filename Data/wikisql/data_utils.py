'''
Author: your name
Date: 1970-01-01 01:00:00
LastEditTime: 2020-09-07 16:03:45
LastEditors: Please set LastEditors
Description: Generate sqls from json file for wikisql
FilePath: /Tree2Seq/Data/wikisql/generate_sql.py
'''
from typing import List, Dict, Tuple
import json
from Data.utils import build_up_vocab, rm_contents_in_brackets, get_flatten_data, build_vocab, pad, get_relations, build_graphs_and_depths, pad_up_to_down_masks, get_single_graph_data, build_graph
from Utils.tree import TreeNode
from Utils.const import WIKISQL_AGG_OPS, WIKISQL_COND_OPS
import numpy as np
from Data.vocab import Vocabulary
import logging


def replace_question(question: str, val_map: Dict) -> str:
    """Replace value into placeholder symbol.

    Args:
        question (str): question
        val_map (Dict): mapping dict recording the mapping

    Returns:
        str: processed question
    """
    # version 1: replace directly
    # version 2:
    #   * judge count first
    #   * replace
    # tokenizer = TweetTokenizer()

    for val in sorted(val_map.keys(), key=lambda x: len(x), reverse=True):
        to_val = val_map[val]
        val = val.lower()
        if question.count(val) == 1:
            question = question.replace(f"\"{val}\"", f" {to_val} ")
            question = question.replace(f"'{val}'", f" {to_val} ")
            question = question.replace(val, f" {to_val} ")

        else:
            temp = question.split()
            while val in temp:
                idx = temp.index(val)
                temp[idx] = f" {to_val} "
            question = " ".join(temp)

    return question


def normalize_tok(tok):
    toks = []
    tok = tok.strip()
    flag = True
    # special case, xxx(xx)
    if tok and tok[-1] == ')' and tok[0] != '(' and '(' in tok:
        idx = tok.index('(')
        toks.append(tok[:idx])
        tok = tok[idx:]
        flag = False
    if "(s" in tok and ")" not in tok:
        tok = tok.replace("(s", "")
    tail = None
    if tok and tok[0] in ['(', '#', ',', '%', ')']:
        toks.append(tok[0])
        tok = tok[1:]
        flag = False

    if tok and tok[-1] in [')', '#', ',', '%']:
        tail = tok[-1]
        tok = tok[:-1]
        flag = False

    tok = tok.strip()
    if tok:
        if tok[0] == '$' and ',' in tok:
            tok = tok.replace(",", "")
        if tok[0] == '$':
            toks.append(tok[0])
            tok = tok[1:]
            flag = False

        if tok:
            if flag:
                tok = tok.replace(",", "")
                toks.append(tok)
            else:
                toks += normalize_tok(tok)
    if tail is not None:
        toks.append(tail)

    return toks


def filter_tok(toks):
    filtered_val = set([])
    new_toks = []
    for tok in toks:
        if tok not in filtered_val:
            new_toks.append(tok)
    return new_toks


def clean_question(question, tokenize=True):
    question = question.lower().strip('"')
    question = question.strip("'")

    # remove brackets
    question = rm_contents_in_brackets(question, "()")
    question = rm_contents_in_brackets(question, "[]")
    question = rm_contents_in_brackets(question, "{}")

    # punctuation: , ? . !
    punctuations = [',', '?', '!']
    for punctuation in punctuations:
        question = question.replace(punctuation, f" {punctuation} ")
    # special .
    if question[-1] == '.':
        question = question[:-1] + " ."

    # 's and s'
    question = question.replace("'s", " 's ")
    question = question.replace("s'", "s ' ")

    toks = question.split()

    # clean toks
    new_toks = []
    idx = 0
    length = len(toks)
    while idx < length:
        tok = toks[idx]
        tok = tok.strip()
        if tok not in ["'s", "'"]:
            tok = tok.strip('"')
            tok = tok.strip("'")
        if not tok:
            idx += 1
            continue

        new_toks += normalize_tok(tok)
        idx += 1

    # filter
    new_toks = filter_tok(new_toks)
    if tokenize:
        return new_toks
    else:
        return " ".join(new_toks)


def clean_column(column, tokenize=True):
    column = column.lower()

    # remove brackets
    column = rm_contents_in_brackets(column, "()")
    column = rm_contents_in_brackets(column, "[]")
    column = rm_contents_in_brackets(column, "{}")

    # double and single quotes and some special symbols
    column = column.strip("'")
    column = column.strip('"!')
    column = column.replace("'s", " 's ")
    column = column.replace("s'", "s ' ")

    # special cases
    # case 1: : to explain, ex: power demand: 40 kw
    column = column.split(":")[0]
    toks = column.split()

    # case 2: smaller power power demand
    if len(toks) > 2 and toks[1] == toks[2]:
        toks = [toks[0]] + toks[2:]

    # case 3: long columns, specifically, including some explainations
    if "print resolution" in column:
        toks = toks[:3]
    elif "critical altitude" in column:
        toks = toks[:2]
    elif "akimel a-al" in column:
        toks = toks[:1]
    elif "birth name as announced" in column:
        toks = toks[:6]
    elif "location chernobyl" in column:
        toks = toks[:1]
    elif "episode no. episode no." in column:
        toks = toks[:2]
    elif "jamie and johns guest dermot" in column:
        toks = toks[:4]
    elif "andrew and georgies guest lee" in column:
        toks = toks[:15]

    # clean tokens
    new_toks = []
    for tok in toks:
        # strip
        tok = tok.strip('"')
        tok = tok.strip("'")
        if not tok:
            continue
        new_toks += normalize_tok(tok)

    # filter
    new_toks = filter_tok(new_toks)

    if not new_toks:
        new_toks = ["&&"]

    if tokenize:
        return new_toks
    else:
        return " ".join(new_toks)


def load_tables(table_file: str) -> Dict[str, List[str]]:
    """Load tables (columns specifically) from table file.

    Args:
        table_file (str): table file path.

    Returns:
        Dict[str, List[str]]: a dict, the key is table_id and value is a column list.
    """
    with open(table_file, "r") as inf:
        tables = inf.read().strip().split("\n")

    table_dict = dict()
    for table in tables:
        table = json.loads(table)
        table_id = table["id"]
        table_columns = table["header"]
        table_dict[table_id] = table_columns
    return table_dict


def get_sqls_and_questions(data_file: str,
                           table_file: str) -> Tuple[List[Dict], List[Dict]]:
    """Load samples (json format) from data file and get the normalized sqls and
    questions (dict format). The questions are lexicalized and both sqls and
    questions are cleaned.

    Args:
        data_file (str): data file path.
        table_file (str): table file path.

    Return:
        Tuple[List[Dict], List[Dict]]: samples list and value map list
    """
    tables = load_tables(table_file)

    with open(data_file, "r") as inf:
        data = inf.read().strip().split("\n")

    samples = []
    val_map_list = []
    for index, sample in enumerate(data):
        # prepare all the variables
        sample = json.loads(sample)
        proc_sample = dict()
        val_map = dict()
        proc_sql = dict()

        # fetch data
        origin_sql = sample["sql"]
        table_id = sample["table_id"]
        conds = origin_sql["conds"]
        table_cols = tables[table_id]
        sel_col = table_cols[origin_sql["sel"]]
        agg = WIKISQL_AGG_OPS[origin_sql["agg"]]
        origin_ques = sample["question"].lower()

        # process data, clean and lower
        # select clause
        proc_sql["sel"] = clean_column(sel_col)
        proc_sql["agg"] = agg
        proc_sql["cond"] = []
        # where clause
        for cond in conds:
            cond_col_idx, cond_op_idx, cond_val = cond
            cond_val = str(cond_val)

            if cond_val not in val_map:
                val_map[cond_val] = f"value_{chr(ord('a') + len(val_map))}"
            cond_col = clean_column(table_cols[cond_col_idx])
            cond_op = WIKISQL_COND_OPS[cond_op_idx]
            cond_val = val_map[cond_val]
            proc_sql["cond"].append([cond_col, cond_op, cond_val])

        # question
        proc_ques = replace_question(origin_ques, val_map)
        proc_ques = clean_question(proc_ques)

        # save data
        proc_sample["sql"] = proc_sql
        proc_sample["origin_ques"] = origin_ques
        proc_sample["proc_ques"] = proc_ques
        proc_sample["table_id"] = table_id

        samples.append(proc_sample)
        val_map_list.append(val_map)

    return samples, val_map_list


def get_cond_tree(cond: List) -> TreeNode:
    col, op, val = cond
    op_node = TreeNode(op, type="condition")
    column = TreeNode("column", parent=op_node, type="column")
    val_node = TreeNode(val, parent=op_node, copy_mark=1, type="token")

    for col_tok in col:
        col_tok_node = TreeNode(col_tok,
                                parent=column,
                                copy_mark=1,
                                type="token")
        column.add_child(col_tok_node)

    op_node.add_children([column, val_node])
    return op_node


def get_trees(sqls: List[Dict]) -> List[TreeNode]:
    """Transform all sqls to trees.

    Args:
        sqls (List[Dict]): sqls dict list and each sql is returned by [get_sqls_and_questions] function.

    Returns:
        List[TreeNode]: trees list, each item is the root node.
    """
    trees = []

    for sql in sqls:
        # construct a tree
        # root node
        root = TreeNode("root", type="root")
        # select node
        select = TreeNode("SelectClause", root, type="clause")
        root.add_child(select)
        # select column token nodes
        sel_col_toks = []
        for sel_col_tok in sql["sel"]:
            sel_col_toks.append(
                TreeNode(sel_col_tok, copy_mark=1, type="token"))

        column = TreeNode("column", select, copy_mark=0, type="column")
        select.add_child(column)
        # agg node
        agg_ = sql["agg"]
        if agg_:
            agg = TreeNode(agg_, column, type="agg")
            column.add_child(agg)

        for sel_col_tok in sel_col_toks:
            sel_col_tok.set_parent(column)
        column.add_children(sel_col_toks)

        conds = sql["cond"]
        if conds:
            # where clause
            where = TreeNode("WhereClause", root, type="clause")
            root.add_child(where)

            for cond in conds:
                cond_root = get_cond_tree(cond)
                where.add_child(cond_root)
                cond_root.set_parent(where)

        trees.append(root)
    return trees


def load_wikisql_data(
    data_files: List[str],
    table_file: str,
    down_vocab: Vocabulary = None,
    up_vocab=None,
    min_freq: int = 1,
    max_depth: int = 4
) -> Tuple[List[List], List[List], np.ndarray, np.ndarray, List[List],
           List[str], Vocabulary, List[Dict], int]:
    """Load all the data required by the model.

    Args:
        data_file (str): data file path.
        table_file (str): table file path.
        vocab (str): vocabulary.
        min_freq (int): the minimum frequency of words in vocabulary.

    Returns:
        Tuple[List[List], List[List], np.ndarray, np.ndarray, List[List], List[str], Vocabulary, List[Dict], int]:
        sqls, questions, relations, cliques, copy_masks, origin_questions, vocabulary, val_map_list, relation_num
    """
    # load data from original files
    logging.info("Start loading data from origin files.")
    samples, val_map_list = get_sqls_and_questions(data_files[0], table_file)

    logging.info("Start constructing trees.")
    # construct trees
    trees = get_trees(list(map(lambda sample: sample["sql"], samples)))

    logging.info("Start getting flatten data.")
    # get flatten sqls, copy_masks, cliques, origin questions, processed questions
    down_nodes, up_nodes, up_nodes_types, down_nodes_types, up_graphs, up_depths, up_schemas, copy_masks, mixed_head_masks, up_to_down_masks, down_to_up_masks = get_flatten_data(
        trees, use_schema=False)
    origin_ques = list(map(lambda sample: sample["origin_ques"], samples))
    proc_ques = list(map(lambda sample: sample["proc_ques"], samples))

    if down_vocab is None:
        logging.info("Start building vocabulary.")
        # build vocabulary
        down_vocab = build_vocab(down_nodes, proc_ques, min_freq=min_freq)
        up_vocab = build_up_vocab(up_nodes, min_freq)

    logging.info("Start post processing, such as padding.")
    # pad sqls, questions, copy_masks and transform them to idx representation
    down_nodes, src2trg_map_list, idx2tok_map_list, tok2idx_map_list = down_vocab.to_ids_2_dim(
        down_nodes, unk=True)
    up_nodes = up_vocab.to_ids_2_dim(up_nodes)
    up_graphs, up_depths = build_graphs_and_depths(up_graphs, up_depths,
                                                   len(up_nodes[0]), max_depth)
    questions = down_vocab.to_ids_2_dim(proc_ques,
                                        add_end=True,
                                        TOK2IDX_MAP_LIST=tok2idx_map_list)
    down_nodes_num = len(down_nodes[0])
    up_nodes_num = len(up_nodes[0])
    copy_masks = pad(copy_masks, max_len=down_nodes_num, pad_val=0)
    up_nodes_types = pad(up_nodes_types, max_len=up_nodes_num, pad_val=0)
    down_nodes_types = pad(down_nodes_types, max_len=down_nodes_num, pad_val=0)
    up_to_down_masks = pad_up_to_down_masks(up_to_down_masks, up_nodes_num,
                                            down_nodes_num)
    down_to_up_masks = pad_up_to_down_masks(down_to_up_masks, down_nodes_num,
                                            up_nodes_num)

    logging.info("Start getting relations")
    down_to_up_relations = get_relations(trees, down_nodes_num, up_nodes_num)
    # relations = np.array(relations)
    # relations = np.expand_dims(relations[:, :, :, 0], axis=-1)
    logging.info("Data has been loaded successfully.")
    return down_nodes, up_nodes, up_nodes_types, down_nodes_types, up_depths, up_schemas, down_to_up_relations, questions, copy_masks, src2trg_map_list, origin_ques, down_vocab, up_vocab, val_map_list, idx2tok_map_list, up_to_down_masks, down_to_up_masks


def get_sequences(
        sqls: List[Dict]) -> Tuple[List[List], List[List], List[List]]:
    """Get sql, clique and copy mask sequences from sql dict.

    Args:
        sqls (List[Dict]): sqls dict.

    Returns:
        Tuple[List[List], List[List], List[List]]: (sql, clique, copy mask)
    """
    sql_seq, cliques, copy_masks = [], [], []

    for sql in sqls:
        sql_toks = []
        clique = []
        copy_mask = []

        cur_clique = 0

        # select clause
        sql_toks.append('select')
        clique.append(0)
        copy_mask.append(0)

        if sql['agg']:
            sql_toks.append(sql['agg'])
            clique.append(cur_clique)
            copy_mask.append(0)

        sql_toks += sql["sel"]
        clique += [cur_clique] * len(sql["sel"])
        copy_mask += [1] * len(sql["sel"])

        cur_clique += 1
        # where clause
        sql_toks.append('where')
        clique.append(cur_clique)
        copy_mask.append(0)
        conds = sql['cond']

        for index, cond in enumerate(conds):
            col_toks, cond_op, val = cond
            sql_toks += col_toks
            sql_toks += [cond_op, val]
            clique += [cur_clique] * len(col_toks)
            copy_mask += [1] * len(col_toks)
            clique += [cur_clique] * 2
            copy_mask += [0, 1]

            if index < len(conds) - 1:
                sql_toks.append('and')
                copy_mask.append(0)
                clique.append(cur_clique)

            # cur_clique += 1

        sql_seq.append(sql_toks)
        cliques.append(clique)
        copy_masks.append(copy_mask)

    return sql_seq, cliques, copy_masks


def load_wikisql_seq2seq_data(data_files: List[str],
                              table_file,
                              vocab: Vocabulary = None,
                              min_freq: int = 1):
    # load data from original files
    logging.info("Start loading data from origin files.")
    samples, val_map_list = get_sqls_and_questions(data_files[0], table_file)
    logging.info("Start constructing sequences.")
    sqls, cliques, copy_masks = get_sequences(
        list(map(lambda sample: sample["sql"], samples)))
    origin_ques = list(map(lambda sample: sample["origin_ques"], samples))
    proc_ques = list(map(lambda sample: sample["proc_ques"], samples))

    if vocab is None:
        logging.info("Start building vocabulary.")
        # build vocabulary
        vocab = build_vocab(sqls, proc_ques, min_freq=min_freq)

    logging.info("Start post processing, such as padding.")
    # pad sqls, questions, copy_masks and transform them to idx representation
    sqls, src2trg_map_list, idx2tok_map_list, tok2idx_map_list = vocab.to_ids_2_dim(
        sqls, unk=True)
    questions = vocab.to_ids_2_dim(proc_ques,
                                   add_end=True,
                                   TOK2IDX_MAP_LIST=tok2idx_map_list)
    node_num = len(sqls[0])
    copy_masks = pad(copy_masks, max_len=node_num, pad_val=0)

    logging.info("Data has been loaded successfully.")
    return sqls, questions, copy_masks, origin_ques, vocab, val_map_list, src2trg_map_list, idx2tok_map_list


def load_wikisql_single_graph_data(data_files: List[str],
                                   table_file,
                                   vocab: Vocabulary = None,
                                   min_freq: int = 1):
    # load data from original files
    logging.info("Start loading data from origin files.")
    samples, val_map_list = get_sqls_and_questions(data_files[0], table_file)

    logging.info("Start constructing trees.")
    # construct trees
    trees = get_trees(list(map(lambda sample: sample["sql"], samples)))

    logging.info("Start getting flatten data.")
    nodes, types, graphs, copy_masks = get_single_graph_data(trees)

    origin_ques = list(map(lambda sample: sample["origin_ques"], samples))
    proc_ques = list(map(lambda sample: sample["proc_ques"], samples))

    if vocab is None:
        logging.info("Start building vocabulary.")
        # build vocabulary
        vocab = build_vocab(nodes, proc_ques, min_freq=min_freq)

    logging.info("Start post processing, such as padding.")
    # pad sqls, questions, copy_masks and transform them to idx representation
    nodes, src2trg_map_list, idx2tok_map_list, tok2idx_map_list = vocab.to_ids_2_dim(
        nodes, unk=True)
    logging.info("nodes has been padded.")
    questions = vocab.to_ids_2_dim(proc_ques,
                                   add_end=True,
                                   TOK2IDX_MAP_LIST=tok2idx_map_list)
    logging.info("question has been padded.")
    node_num = len(nodes[0])
    logging.info(f"maximum node number: {node_num}")
    copy_masks = pad(copy_masks, max_len=node_num, pad_val=0)
    logging.info("copy mask has been padded.")

    types = pad(types, max_len=node_num, pad_val=0)
    logging.info("type has been padded.")

    graphs = build_graph(graphs, node_num)
    logging.info("graph has been padded.")

    logging.info("Data has been loaded successfully.")

    return nodes, types, questions, graphs, copy_masks, origin_ques, vocab, val_map_list, src2trg_map_list, idx2tok_map_list


# TODO
def load_wikisql_tree_data():
    pass
