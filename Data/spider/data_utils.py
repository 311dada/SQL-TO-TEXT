'''
Author: your name
Date: 1970-01-01 01:00:00
LastEditTime: 2020-09-07 16:12:23
LastEditors: Please set LastEditors
Description: Generate sqls from json file for wikisql
FilePath: /Tree2Seq/Data/wikisql/generate_sql.py
'''
from typing import List, Dict, Tuple
from Utils.tree import TreeNode
from Data.vocab import Vocabulary
import numpy as np
import json
import re
from Data.utils import rm_contents_in_brackets, get_flatten_data, build_vocab, pad, get_relations, build_graphs_and_depths, get_up_schemas, pad_up_to_down_masks, build_up_vocab
from nltk.tokenize import TweetTokenizer
from Utils.const import SPIDER_MONTH, SPIDER_MAP
from Data.spider.parse import get_schemas_from_json, Schema
from Data.spider.contruct_tree import build_tree
from Data.spider.generate import get_normalized_sql
import logging


def load_dbs(table_file: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Load databases (tables and columns specificly) from table file.

    Args:
        table_file (str): table file path.

    Returns:
        Dict[str, Dict[str, Dict[str, str]]]: a dict, the key is db_id and
        value is is another dict in the following format.

        key: value, key is among ['tables', 'columns'] and value is the mapping
        dict for tables or columns abount the original name and real name.
    """
    with open(table_file, "r") as f:
        tables = json.load(f)

    tables_map = dict()
    for table in tables:
        db_id = table["db_id"]
        tables_map[db_id] = dict()
        tables_map[db_id]["tables"] = dict()
        tables_map[db_id]["columns"] = dict()

        # table mapping
        for index, origin_table_name in enumerate(
                table["table_names_original"]):
            tables_map[db_id]["tables"][origin_table_name.lower(
            )] = table["table_names"][index].lower()

        # column mapping
        original_tables = table["table_names_original"]
        for index, origin_col_name in enumerate(
                table["column_names_original"]):
            if index == 0:
                continue
            original_column_name = origin_col_name[1].lower()
            original_table = original_tables[origin_col_name[0]].lower()
            if original_table not in tables_map[db_id]["columns"]:
                tables_map[db_id]["columns"][original_table] = dict()
            tables_map[db_id]["columns"][original_table][
                original_column_name] = table["column_names"][index][1].lower(
                )
        table_names_original = table["table_names_original"]
        column_names_original = table["column_names_original"]
        # primary keys
        primary_keys = table["primary_keys"]
        tables_map[db_id]["primary"] = set()
        for primary_key in primary_keys:
            table_ = table_names_original[column_names_original[primary_key]
                                          [0]].lower()
            column = column_names_original[primary_key][1].lower()
            tables_map[db_id]["primary"].add(f"{table_}.{column}")

        # foreign keys
        foreign_keys = table["foreign_keys"]
        tables_map[db_id]["foreign"] = set()

        for foreign_key in foreign_keys:
            table1 = table_names_original[column_names_original[foreign_key[0]]
                                          [0]].lower()
            column1 = column_names_original[foreign_key[0]][1].lower()
            tab_col_1 = f"{table1}.{column1}"

            table2 = table_names_original[column_names_original[foreign_key[1]]
                                          [0]].lower()
            column2 = column_names_original[foreign_key[1]][1].lower()
            tab_col_2 = f"{table2}.{column2}"

            tables_map[db_id]["foreign"].add((tab_col_1, tab_col_2))

    return tables_map


def normalize_query(query: List[str]) -> List[str]:
    """Normalize queries with and without values.
    Ex. split T1.col1 to T1, ., col1.

    Args:
        query (List[str]): query token list.

    Returns:
        List[str]: normalized query token list.
    """
    temp: List = []
    index: int = 0
    length: int = len(query)
    while index < length:
        sql_val = query[index]
        if sql_val == "``":
            start = index
            index += 1
            while query[index] != "''" and query[index] != "``":
                index += 1
            temp += [" ".join(query[start:index + 1])]

        else:
            if sql_val[0].lower() == 't':
                temp += re.split(r"([.])", sql_val)
            else:
                temp += [sql_val]
        index += 1
    return temp


def join_val(toks: List[str]) -> str:
    """Join value token list to a value token.

    Args:
        toks (List[str]): value token list.

    Returns:
        str: value token.
    """
    val = ""
    for index, tok in enumerate(toks):
        if index == 0:
            val += tok
        else:
            pre = toks[index - 1]
            if pre in [".", "%", "@"] or tok in [".", "%", "@"]:
                val += tok
            else:
                val += " " + tok
    return val


def get_val_map_list(data_files: List[str]) -> List[Dict]:
    """Get value map list from data files.

    Args:
        data_files (List[str]): data files path.

    Returns:
        List[Dict]: value map list.
    """
    # do not process limit 1
    val_mapping_list = []
    for data_path in data_files:
        with open(data_path, "r") as f:
            data = json.load(f)

        for sample in data:
            val_map = dict()
            query_toks = sample["query_toks"]
            query_toks_no_value = sample["query_toks_no_value"]

            sql_with_value = normalize_query(query_toks)
            sql_no_value = normalize_query(query_toks_no_value)

            i, j = 0, 0
            with_val, no_val = len(sql_with_value), len(sql_no_value)

            while i < with_val and j < no_val:
                if sql_with_value[i].lower() == sql_no_value[j].lower():
                    i += 1
                    j += 1
                elif sql_no_value[j].lower() == "value":
                    buf = j
                    start = i
                    if j == no_val - 1:
                        i = with_val
                    else:
                        next_ = sql_no_value[j + 1].lower()
                        while i < with_val and sql_with_value[i].lower(
                        ) != next_:
                            i += 1
                    val = join_val(sql_with_value[start:i])
                    j += 1
                    if buf > 0 and sql_no_value[buf - 1] == "limit":
                        continue
                    val_ = val.lower().strip("' `;")
                    val_ = join_val(val_.split())
                    if val_ == "comp.sci.":
                        val_ = "comp. sci."
                    elif val_ == "computer info.systems":
                        val_ = "computer info. systems"
                    if val_ not in val_map:
                        val_map[val_] = f"value_{chr(ord('a') + len(val_map))}"
            val_mapping_list.append(val_map)
    return val_mapping_list


def is_float(tok: str) -> bool:
    """Judge if the tok is a float.

    Args:
        tok (str): token.

    Returns:
        bool: whether is a float.
    """
    try:
        _ = float(tok)
        return True
    except ValueError:
        return False


def is_int(tok: str) -> bool:
    """Judge if the tok is an int.

    Args:
        tok (str): token.

    Returns:
        bool: whether is an int.
    """
    try:
        _ = int(tok)
        return True
    except ValueError:
        return False


def proc_val(toks: List[str]) -> str:
    """Process value list.

    Args:
        toks (List[str]): token list.

    Returns:
        str: value.
    """
    index = 0
    val = ""
    length = len(toks)

    while index < length:
        tok = toks[index]
        if index < length - 1:
            next_tok = toks[index + 1]
            if next_tok in ",!?:":
                val = val + tok
            else:
                val = val + tok + " "
        else:
            val = val + tok
        index += 1
    return val


def lexicalize_question(sent: str, val_map: Dict) -> str:
    """Lexicalize the question.

    Args:
        sent (str): question.
        val_map (Dict): value map dict.

    Returns:
        str: lexicalized question.
    """
    sent = sent.lower()
    sent = sent.replace('"', "")
    for val in sorted(val_map.keys(), key=lambda x: -len(x)):
        to_rep = " " + val_map[val] + " "

        val_ = val.lower().strip("' `;%/:")
        if not val_:
            continue
        val_ = join_val(val_.split())
        if val_ == "comp.sci.":
            val_ = "comp. sci."
        elif val_ == "computer info.systems":
            val_ = "computer info. systems"

        double_quote_val = '"' + val_ + '"'
        single_quote_val = "'" + val_ + "'"
        normal_val = val_.strip("%/")
        double_quote_normal_val = '"' + normal_val + '"'
        single_quote_normal_val = "'" + normal_val + "'"
        if double_quote_val in sent:
            sent = sent.replace(double_quote_val, to_rep)
        elif single_quote_val in sent:
            sent = sent.replace(single_quote_val, to_rep)
        elif double_quote_normal_val in sent:
            sent = sent.replace(double_quote_normal_val, to_rep)
        elif single_quote_normal_val in sent:
            sent = sent.replace(single_quote_normal_val, to_rep)
        elif is_float(val_):
            val__ = str(float(val_))
            if val__ in sent:
                sent = sent.replace(val__, to_rep)
            elif is_int(val_):
                val__ = str(int(val_))
                cnt = sent.count(val__)

                if cnt == 1:
                    sent = sent.replace(val__, to_rep)
                elif cnt > 1:
                    sent_toks = sent.split()
                    for idx, tok in enumerate(sent_toks):
                        if tok == val__:
                            sent_toks[idx] = to_rep
                    sent = " ".join(sent_toks)
                elif 0 < int(val_) < len(SPIDER_MONTH):
                    month = SPIDER_MONTH[int(val_)]
                    sent = sent.replace(month, to_rep)

        elif val_ in sent or normal_val in sent:
            sent = sent.replace(val_, to_rep)
            sent = sent.replace(normal_val, to_rep)
        else:
            # specical case
            if val_ in SPIDER_MAP:
                val_ = SPIDER_MAP[val_]

            elif val_ == "tony award":
                val_ = "bob"

            else:
                val_ = val_.split()
                val_ = proc_val(val_)
                if val_ in sent:
                    sent = sent.replace(val_, to_rep)

            if isinstance(val_, str):
                sent = sent.replace(val_, to_rep)
            else:
                for v_ in val_:
                    sent = sent.replace(v_, to_rep)
                    sent = sent.replace("'" + v_ + "'", to_rep)
                    sent = sent.replace('"' + v_ + '"', to_rep)

    return " ".join(sent.strip().split())


def clean_question(ques: str) -> List[str]:
    """Remove () and deal with 's and s'.

    Args:
        ques (str): question.

    Returns:
        str: clean question.
    """
    ques = ques.lower()
    ques = rm_contents_in_brackets(ques)
    ques = ques.replace("'s", " 's ")
    ques = ques.replace("s'", "s ' ")
    ques = ques.replace("'t", " 't ")
    tokenizer = TweetTokenizer()
    return tokenizer.tokenize(ques)


def normalize_val(val: str) -> str:
    """Return a normalized value.

    Args:
        val (str): value.

    Returns:
        str: normalized value.
    """
    val = val.strip('"')
    val = val.strip("'")
    return "".join(val.split())


def normalize_sql(sql: str) -> str:
    """Normalize value in sql.

    Args:
        sql (str): sql.

    Returns:
        str: normalized sql.
    """

    index = 0
    length = len(sql)
    pre = None
    normalized_sql = ""
    while index < length:
        tok = sql[index]
        if tok == "'" or tok == '"':
            if pre is None:
                pre = tok
                index += 1
                continue
            elif pre == tok:
                pre = None
                index += 1
                continue
        if pre is None:
            normalized_sql = normalized_sql + tok
        elif tok != " ":
            normalized_sql = normalized_sql + tok

        index += 1
    return normalized_sql


def lexicalize_sql(sql: str, val_map: Dict, index=0) -> str:
    """Lexicalize the sql.

    Args:
        sql (str): sql.
        val_map (Dict): value map dict.

    Returns:
        str: lexicalized sql.
    """
    # naive version, replace directly
    sql = sql.lower().strip(';')
    for val in sorted(val_map.keys(), key=lambda x: len(x), reverse=True):
        to_val = val_map[val]
        if f"\"{val}\"" in sql:
            sql = sql.replace(f"\"{val}\"", f" {to_val} ")
        elif f"'{val}'" in sql:

            sql = sql.replace(f"'{val}'", f" {to_val} ")

        else:
            sql = normalize_sql(sql)
            val = normalize_val(val)
            cnt = sql.count(val)
            if cnt == 1:
                sql = sql.replace(val, f" {to_val} ")

            elif cnt > 1:
                temp = sql.split()
                while val in temp:
                    idx = temp.index(val)
                    temp[idx] = f" {to_val} "
                sql = " ".join(temp)
    return " ".join(sql.strip().split()).strip()


def get_sqls_and_questions(data_files: List[str],
                           table_file: str) -> Tuple[List[Dict], List[Dict]]:
    """Load samples (json format) from data files and get normalized sqls and questions (dict format). The questions are lexicalized and both sqls and questions are cleanned.

    Args:
        data_files (List[str]): data file paths.
        table_file (str): table file path.

    Returns:
        Tuple[List[Dict], List[Dict]]: samples list and value map list.
    """
    # get value map list
    val_map_list = get_val_map_list(data_files)

    # get samples list
    # lexicalized sqls (token), original questions, lexicalized questions (token list)
    """
        sqls:
            1. lexicalize original sql query
            2. lowercase and rm tail ;
        questions:
            1. save original questions: clean "
            2. lexicalize questions
            3. clean questions
                a) lowercase
                b) 's and s'
                c) remove ()
    """
    # load data
    data = []
    for data_file in data_files:
        with open(data_file, "r") as inf:
            data = data + json.load(inf)

    sample_list = []
    for index, sample in enumerate(data):
        proc_sample = dict()
        # get processed sample dict
        db_id = sample["db_id"]
        proc_sample["db_id"] = db_id

        val_map = val_map_list[index]

        # sqls
        origin_sql = sample["query"]
        lexi_sql = lexicalize_sql(origin_sql, val_map, index)
        proc_sample["sql"] = lexi_sql

        # quesitons
        origin_ques = sample["question"].lower()
        proc_sample["origin_ques"] = origin_ques
        # origin_ques = origin_ques.replace('"', "")

        lexi_ques = lexicalize_question(sample["question"], val_map)

        lexi_ques = clean_question(lexi_ques)
        proc_sample["proc_ques"] = lexi_ques

        sample_list.append(proc_sample)

    return sample_list, val_map_list


def get_trees(sqls: List[str], dbs: List[str],
              tables_map: Dict[str, Dict[str, Dict[str, str]]],
              table_file: str) -> List[TreeNode]:

    # construct trees, implement it in construct tree py recursively.
    schemas, db_names, tables = get_schemas_from_json(table_file)

    ans = []

    for index, sql in enumerate(sqls):
        db_id = dbs[index]
        schema = schemas[db_id]
        table = tables[db_id]
        schema = Schema(schema, table)
        table_map = tables_map[db_id]
        root = build_tree(sql, table_map, schema)
        ans.append(root)
    return ans


def get_mapping_name(tok: str, table_map: Dict[str, Dict[str, Dict[str,
                                                                   str]]]):
    # table case
    if "." not in tok:
        table = table_map["tables"][tok]
        table_toks = table.split()
        copy_mask = [1] * len(table_toks)
        return table_toks, copy_mask

    else:
        table, column = tok.split(".")
        table_toks = table_map["tables"][table].split()
        column_toks = table_map["columns"][table][column].split()
        mapping_toks = table_toks + ["."] + column_toks
        copy_mask = [1] * len(table_toks) + [0] + [1] * len(column_toks)
        return mapping_toks, copy_mask


def get_sequences(sqls: List[str], dbs: List[str],
                  tables_map: Dict[str, Dict[str, Dict[str, str]]],
                  table_file: str):
    sql_seq, cliques, copy_masks = [], [], []
    schemas, db_names, tables = get_schemas_from_json(table_file)

    for index, sql in enumerate(sqls):
        new_sql_toks = []
        copy_mask = []
        db_id = dbs[index]
        sql = get_normalized_sql(sql, db_id, schemas, tables)
        table_map = tables_map[db_id]
        sql_toks = sql.split()

        for tok in sql_toks:
            if "." not in tok and tok not in table_map["tables"]:
                new_sql_toks.append(tok)
                copy_mask.append(0)
            else:
                # name mapping
                mapping_toks, copy_mask_ = get_mapping_name(tok, table_map)
                new_sql_toks += mapping_toks
                copy_mask += copy_mask_
        sql_seq.append(new_sql_toks)
        copy_masks.append(copy_mask)

    # TODO: cliques
    cliques = copy_masks.copy()

    return sql_seq, cliques, copy_masks


def add_schemas(tables_map, vocab):
    for schema in tables_map.values():
        tables = schema["tables"]

        for table in tables.values():
            vocab.add_words(table.lower().split())

        table_columns = schema["columns"].values()

        for table in table_columns:
            columns = table.values()
            for column in columns:
                vocab.add_words(column.lower().split())
    return vocab


def load_spider_data(
    data_files: List[str],
    table_file: str,
    down_vocab: Vocabulary = None,
    up_vocab: Vocabulary = None,
    min_freq: int = 1,
    max_depth: int = 4
) -> Tuple[List[List], List[List], np.ndarray, np.ndarray, List[List],
           List[str], Vocabulary, List[Dict], int]:
    """Load all the data required by the model.

    Args:
        data_file (List[str]): data files path.
        table_file (str): table file path.
        vocab (str): vocabulary.
        min_freq (int): the minimum frequency of words in vocabulary.

    Returns:
        Tuple[List[List], List[List], np.ndarray, np.ndarray, List[List], List[str], Vocabulary, List[Dict], int]:
        sqls, questions, relations, cliques, copy_masks, origin_questions, vocabulary, val_map_list, relation_num
    """
    # load data from original files
    logging.info("Start loading data from origin files.")
    samples, val_map_list = get_sqls_and_questions(data_files, table_file)

    logging.info("Start constructing trees.")
    tables_map = load_dbs(table_file)
    sqls = list(map(lambda sample: sample["sql"], samples))
    dbs = list(map(lambda sample: sample["db_id"], samples))
    trees = get_trees(sqls, dbs, tables_map, table_file)

    logging.info("Start getting flatten data.")
    # get flatten sqls, copy_masks, cliques, origin questions, processed questions
    down_nodes, up_nodes, up_nodes_types, down_nodes_types, up_graphs, up_depths, up_schemas, copy_masks, mixed_head_masks, up_to_down_masks, down_to_up_masks = get_flatten_data(
        trees, tables_map, dbs, max_depth)

    origin_ques = list(map(lambda sample: sample["origin_ques"], samples))
    proc_ques = list(map(lambda sample: sample["proc_ques"], samples))

    if down_vocab is None:
        logging.info("Start building vocabulary.")
        # build vocabulary
        down_vocab = build_vocab(down_nodes, proc_ques, min_freq=min_freq)
        down_vocab = add_schemas(tables_map, down_vocab)
        up_vocab = build_up_vocab(up_nodes, min_freq)

    logging.info("Start post processing, such as padding.")
    # pad sqls, questions, copy_masks and transform them to idx representation
    down_nodes, src2trg_map_list, idx2tok_map_list, tok2idx_map_list = down_vocab.to_ids_2_dim(
        down_nodes, unk=True)
    up_nodes = up_vocab.to_ids_2_dim(up_nodes)
    up_graphs, up_depths = build_graphs_and_depths(up_graphs, up_depths,
                                                   len(up_nodes[0]), max_depth)
    logging.info("sql has been padded.")
    questions = down_vocab.to_ids_2_dim(proc_ques,
                                        add_end=True,
                                        TOK2IDX_MAP_LIST=tok2idx_map_list)
    logging.info("question has been padded.")
    down_nodes_num = len(down_nodes[0])
    up_nodes_num = len(up_nodes[0])
    logging.info(f"maximum down node number: {down_nodes_num}")
    logging.info(f"maximum up node number: {up_nodes_num}")

    copy_masks = pad(copy_masks, max_len=down_nodes_num, pad_val=0)
    logging.info("copy mask has been padded.")
    up_nodes_types = pad(up_nodes_types, max_len=up_nodes_num, pad_val=0)
    down_nodes_types = pad(down_nodes_types, max_len=down_nodes_num, pad_val=0)
    up_schemas = get_up_schemas(up_schemas, up_nodes_num)
    up_to_down_masks = pad_up_to_down_masks(up_to_down_masks, up_nodes_num,
                                            down_nodes_num)

    down_to_up_masks = pad_up_to_down_masks(down_to_up_masks, down_nodes_num,
                                            up_nodes_num)

    logging.info("Start getting relations")
    down_to_up_relations = get_relations(trees, down_nodes_num, up_nodes_num)

    logging.info("Data has been loaded successfully.")
    return down_nodes, up_nodes, up_nodes_types, down_nodes_types, up_depths, up_schemas, down_to_up_relations, questions, copy_masks, src2trg_map_list, origin_ques, down_vocab, up_vocab, val_map_list, idx2tok_map_list, up_to_down_masks, down_to_up_masks


def load_spider_seq2seq_data(data_files: List[str],
                             table_file: str,
                             vocab: Vocabulary = None,
                             min_freq: int = 1):
    # load data from original files
    logging.info("Start loading data from origin files.")
    samples, val_map_list = get_sqls_and_questions(data_files, table_file)

    logging.info("Start constructing sequences.")
    tables_map = load_dbs(table_file)
    sqls = list(map(lambda sample: sample["sql"], samples))

    dbs = list(map(lambda sample: sample["db_id"], samples))
    sqls, cliques, copy_masks = get_sequences(sqls, dbs, tables_map,
                                              table_file)
    # sqls = list(map(lambda sql: ['CLS'] + sql, sqls))

    origin_ques = list(map(lambda sample: sample["origin_ques"], samples))
    proc_ques = list(map(lambda sample: sample["proc_ques"], samples))

    if vocab is None:
        logging.info("Start building vocabulary.")
        # build vocabulary
        vocab = build_vocab(sqls, proc_ques, min_freq=min_freq)
        vocab = add_schemas(tables_map, vocab)

    logging.info("Start post processing, such as padding.")
    # pad sqls, questions, copy_masks and transform them to idx representation
    sqls, src2trg_map_list, idx2tok_map_list, tok2idx_map_list = vocab.to_ids_2_dim(
        sqls, unk=True)
    logging.info("sql has been padded.")
    questions = vocab.to_ids_2_dim(proc_ques,
                                   add_end=True,
                                   TOK2IDX_MAP_LIST=tok2idx_map_list)
    logging.info("question has been padded.")
    node_num = len(sqls[0])
    logging.info(f"maximum node number: {node_num}")
    copy_masks = pad(copy_masks, max_len=node_num, pad_val=0)
    logging.info("copy mask has been padded.")

    logging.info("Data has been loaded successfully.")
    return sqls, questions, copy_masks, origin_ques, vocab, val_map_list, src2trg_map_list, idx2tok_map_list
