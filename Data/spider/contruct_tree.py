'''
Author: your name
Date: 2020-08-18 21:09:08
LastEditTime: 2020-09-06 21:56:07
LastEditors: Please set LastEditors
Description: construct a tree given a sql
FilePath: /hpc/Data/spider/contruct_tree.py
'''
from Utils.tree import TreeNode
from typing import Dict, List, Tuple
from Data.spider.parse import Schema, get_sql


def get_val_unit_tree(
        val_unit: Tuple,
        table_map: Dict[str, Dict[str, str]]) -> Tuple[TreeNode, int]:
    """Generate a value tree.

    Args:
        val_unit (Dict): value unit.

    Returns:
        TreeNode: the root of value clause subtree.
    """
    unit_op, col_unit1, col_unit2 = val_unit

    if unit_op != "none":
        unit = TreeNode(unit_op, type="unit")
        column1 = get_col_unit_tree(col_unit1, table_map)
        column2 = get_col_unit_tree(col_unit2, table_map)

        unit.add_children([column1, column2])
        column1.set_parent(unit)
        column2.set_parent(unit)
        return unit
    else:
        return get_col_unit_tree(col_unit1, table_map)


def get_val_tree(val, table_map: Dict[str, Dict[str, str]]):
    if isinstance(val, dict):
        root = get_sql_tree(val, table_map)
        return root
    elif isinstance(val, tuple):
        return get_col_unit_tree(val, table_map)
    else:
        return TreeNode(val, type="token", copy_mark=1)


def get_cond_tree(
        cond_unit: Tuple,
        table_map: Dict[str, Dict[str, str]]) -> Tuple[TreeNode, int]:
    not_op, op, val_unit, val1, val2 = cond_unit
    root = TreeNode(op, type="condition")

    if not_op:
        not_ = TreeNode("not", type="not")
        root.add_child(not_)
        not_.set_parent(root)

    if val_unit is not None:
        val_ = get_val_unit_tree(val_unit, table_map)
        root.add_child(val_)
        val_.set_parent(root)

        if op == "between":
            val_1_node = get_val_tree(val1, table_map)
            val_2_node = get_val_tree(val2, table_map)

            val_1_node.set_parent(root)
            val_2_node.set_parent(root)
            root.add_child(val_1_node)
            root.add_child(val_2_node)
        else:
            val_1_node = get_val_tree(val1, table_map)

            val_1_node.set_parent(root)
            root.add_child(val_1_node)
    else:
        val_1_node = get_val_tree(val1, table_map)

        val_1_node.set_parent(root)
        root.add_child(val_1_node)
    return root


def get_cond_list_tree(
        cond_list: List,
        table_map: Dict[str, Dict[str, str]]) -> Tuple[TreeNode, int]:
    """Generate a condition list tree.

    Args:
        cond_list (List): condition list.

    Returns:
        TreeNode: the root of the condition list.
    """

    pre_node = None
    pre_logic = None

    for cond in cond_list:
        if isinstance(cond, str):
            if pre_logic is None or cond != pre_logic:
                logic_node = TreeNode(cond, type="logic")
                logic_node.add_child(pre_node)
                pre_node.set_parent(logic_node)
                pre_logic = cond
                pre_node = logic_node
        else:
            new_cond = get_cond_tree(cond, table_map)
            if pre_node is None:
                pre_node = new_cond
            else:
                pre_node.add_child(new_cond)
                new_cond.set_parent(pre_node)

    return pre_node


def get_split_col_nodes(col,
                        table_map: Dict[str, Dict[str, str]],
                        clique_mark: int = 0,
                        col_mark: int = 0) -> List[TreeNode]:
    toks = col.split(".")
    if len(toks) == 1:
        column = toks[0]
        column_toks = column.split()
    else:
        table, column = toks
        column_toks = table_map["columns"][table][column].split()

    ans = []

    # column
    for column_tok in column_toks:
        if column_tok == "*":
            copy = 0
        else:
            copy = 1
        ans.append(TreeNode(column_tok, copy_mark=copy, type="token"))

    return ans


def get_col_unit_tree(col_unit: Tuple,
                      table_map: Dict[str, Dict[str, str]]) -> TreeNode:
    agg_, col, isDistinct = col_unit

    column = TreeNode("column", type="column", schema=col)

    if isDistinct:
        distinct = TreeNode("distinct", type="distinct")
        column.add_child(distinct)
        distinct.set_parent(column)

    col_nodes = get_split_col_nodes(col, table_map)

    child = col_nodes
    if agg_ != "none":
        agg = TreeNode(agg_, parent=column, type="agg")
        column.add_child(agg)

    if isinstance(child, list):
        column.add_children(child)
        for ch in child:
            ch.set_parent(column)
    else:
        column.add_child(child)
        child.set_parent(column)
    return column


def get_select_tree(
        select_clause: Dict,
        table_map: Dict[str, Dict[str, str]]) -> Tuple[TreeNode, int, int]:
    """Generate a select clause tree.

    Args:
        select_clause (Dict): select clause dict.

    Returns:
        TreeNode: the root of select clause subtree.
    """
    select = TreeNode("SelectClause", type="clause")

    # distinct
    if select_clause[0]:
        distinct = TreeNode("distinct", parent=select, type="distinct")
        select.add_child(distinct)

    # columns
    for index, col in enumerate(select_clause[1]):
        agg_, val_unit = col
        val = get_val_unit_tree(val_unit, table_map)

        parent = val
        if agg_ != "none":
            agg = TreeNode(agg_, parent=parent, type="agg")
            parent.add_child(agg, front=True)
        select.add_child(parent)
        parent.set_parent(select)

    return select


def get_table_split_nodes(
        table_name: str, table_map: Dict[str, Dict[str,
                                                   str]]) -> List[TreeNode]:
    table_toks = table_map["tables"][table_name].split()
    ans = []
    for table_tok in table_toks:
        ans.append(TreeNode(table_tok, type="token", copy_mark=1))
    return ans


def get_table_node(table_name: str, table_map: Dict[str, Dict[str, str]]):
    table = TreeNode("table", type="table", schema=table_name)
    table_toks = get_table_split_nodes(table_name, table_map)

    for table_tok in table_toks:
        table.add_child(table_tok)
        table_tok.set_parent(table)

    return table


def get_from_tree(
        from_clause: Dict,
        table_map: Dict[str, Dict[str, str]]) -> Tuple[TreeNode, int, int]:
    """Generate a from clause tree.

    Args:
        from_clause (Dict): from vlause dict.

    Returns:
        TreeNode: the root of from clause subtree.
    """
    from_ = TreeNode("FromClause", type="clause")
    table_units = from_clause["table_units"]

    if table_units[0][0] == "sql":
        sql = get_sql_tree(table_units[0][1], table_map)
        from_.add_child(sql)
        sql.set_parent(from_)
    else:
        tables = list(map(lambda x: x[1], table_units))

        for table in tables:
            table_node = get_table_node(table, table_map)
            table_node.set_parent(from_)
            from_.add_child(table_node)

        for index, cond_unit in enumerate(from_clause["conds"]):
            if not isinstance(cond_unit, tuple):
                continue

            cond = get_cond_tree(cond_unit, table_map)
            from_.add_child(cond)
            cond.set_parent(from_)
    return from_


def get_where_tree(
        where_clause: Dict,
        table_map: Dict[str, Dict[str, str]]) -> Tuple[TreeNode, int, int]:
    """Generate a where clause tree.

    Args:
        where_clause (Dict): where vlause dict.

    Returns:
        TreeNode: the root of where clause subtree.
    """
    where = TreeNode("WhereClause", type="clause")

    condition = get_cond_list_tree(where_clause, table_map)
    where.add_child(condition)
    condition.set_parent(where)
    return where


def get_group_by_tree(
        groupBy_clause: List,
        table_map: Dict[str, Dict[str, str]]) -> Tuple[TreeNode, int, int]:
    """Generate a group by clause tree.

    Args:
        group by clause (Dict): group by clause dict.

    Returns:
        TreeNode: the root of group by clause subtree.
    """
    if not groupBy_clause:
        return None
    groupBy = TreeNode("GroupByClause", type="clause")

    for index, col_unit in enumerate(groupBy_clause):
        col = get_col_unit_tree(col_unit, table_map)
        groupBy.add_child(col)
        col.set_parent(groupBy)

    return groupBy


def get_having_tree(
        having_clause: List,
        table_map: Dict[str, Dict[str, str]]) -> Tuple[TreeNode, int, int]:
    """Generate a having clause tree.

    Args:
        having (Dict): having clause dict.

    Returns:
        TreeNode: the root of having clause subtree.
    """
    if not having_clause:
        return None
    having = TreeNode("HavingClause", type="clause")

    condition = get_cond_list_tree(having_clause, table_map)
    having.add_child(condition)
    condition.set_parent(having)

    return having


def get_order_by_tree(
        orderBy_clause: List,
        table_map: Dict[str, Dict[str, str]]) -> Tuple[TreeNode, int, int]:
    """Generate an order by clause tree.

    Args:
        orderBy (Dict): order by clause

    Returns:
        TreeNode: the root of order by subtree.
    """
    if not orderBy_clause:
        return None
    orderBy = TreeNode("OrderByClause", type="clause")

    for index, val_unit in enumerate(orderBy_clause[1]):
        val = get_val_unit_tree(val_unit, table_map)
        orderBy.add_child(val)
        val.set_parent(orderBy)

    order = TreeNode(orderBy_clause[0], parent=orderBy, type="order")
    orderBy.add_child(order)
    return orderBy


def get_limit_tree(limit_clause: int) -> TreeNode:
    """Generate a limit clause tree.

    Args:
        limit (Dict): limit clause.

    Returns:
        TreeNode: the root of limit subtree.
    """
    if limit_clause is None:
        return None
    limit = TreeNode("LimitClause", type="clause")
    val = TreeNode(str(limit_clause), parent=limit, type="token")
    limit.add_child(val)
    return limit


def get_sql_tree(
        sql: Dict,
        table_map: Dict[str, Dict[str, str]]) -> Tuple[TreeNode, int, int]:
    """Generate a sql tree from a parsed sql dict. This function also takes the
    responsibility of mapping original table and column names to natural language
    names.

    Args:
        sql (Dict): parsed sql dict.
        table_map (Dict[str, Dict[str, str]]): table map

    Returns:
        TreeNode: the root node of parsed sql tree.
    """
    root = TreeNode("root", type="root")
    lcs = root

    # intersect
    if "intersect" in sql and sql["intersect"] is not None:
        intersect = TreeNode("intersect", parent=root, type="set")
        root.add_child(intersect)
        lcs = intersect
        sql_sub_tree1 = get_sql_tree(sql["intersect"], table_map)
        del sql["intersect"]
        sql_sub_tree2 = get_sql_tree(sql, table_map)
        lcs.add_child(sql_sub_tree1)
        sql_sub_tree1.set_parent(root)
        lcs.add_child(sql_sub_tree2)
        sql_sub_tree2.set_parent(root)

    # union
    if "union" in sql and sql["union"] is not None:
        union = TreeNode("union", parent=root, type="set")
        root.add_child(union)
        lcs = union
        sql_sub_tree1 = get_sql_tree(sql["union"], table_map)
        del sql["union"]
        sql_sub_tree2 = get_sql_tree(sql, table_map)
        lcs.add_child(sql_sub_tree1)
        sql_sub_tree1.set_parent(root)
        lcs.add_child(sql_sub_tree2)
        sql_sub_tree2.set_parent(root)

    # except
    if "except" in sql and sql["except"] is not None:
        except_ = TreeNode("except", parent=root, type="set")
        root.add_child(except_)
        lcs = except_
        sql_sub_tree1 = get_sql_tree(sql["except"], table_map)
        del sql["except"]
        sql_sub_tree2 = get_sql_tree(sql, table_map)
        lcs.add_child(sql_sub_tree1)
        sql_sub_tree1.set_parent(root)
        lcs.add_child(sql_sub_tree2)
        sql_sub_tree2.set_parent(root)

    # select
    select = get_select_tree(sql["select"], table_map)
    if select is not None:
        lcs.add_child(select)
        select.set_parent(lcs)

    # from
    from_ = get_from_tree(sql["from"], table_map)
    if from_ is not None:
        lcs.add_child(from_)
        from_.set_parent(lcs)

    # where
    if sql["where"]:
        where = get_where_tree(sql["where"], table_map)
        if where is not None:
            lcs.add_child(where)
            where.set_parent(lcs)

    # group by
    group_by = get_group_by_tree(sql["groupBy"], table_map)
    if group_by is not None:
        lcs.add_child(group_by)
        group_by.set_parent(lcs)

    # having
    having = get_having_tree(sql["having"], table_map)
    if having is not None:
        lcs.add_child(having)
        having.set_parent(lcs)

    # order by
    order_by = get_order_by_tree(sql["orderBy"], table_map)
    if order_by is not None:
        lcs.add_child(order_by)
        order_by.set_parent(lcs)

    # limit
    limit = get_limit_tree(sql["limit"])
    if limit is not None:
        lcs.add_child(limit)
        limit.set_parent(lcs)

    return root


def build_tree(sql: str, table_map: Dict[str, Dict[str, str]],
               schema: Schema) -> TreeNode:
    """Build the sql tree from sql string.

    Args:
        sql (str): sql string.
        table_map (Dict[str, Dict[str, str]]): table map.
        schema (Schema): schema

    Returns:
        TreeNode: the root of the sql tree.
    """
    # parse sql
    parsed_sql = get_sql(schema, sql)

    # build tree
    root = get_sql_tree(parsed_sql, table_map)

    return root
