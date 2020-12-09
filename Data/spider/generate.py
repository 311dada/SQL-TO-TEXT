'''
Author: your name
Date: 2020-08-07 13:05:35
LastEditTime: 2020-09-06 21:17:45
LastEditors: Please set LastEditors
Description: generate normalized sql
FilePath: /Tree2Seq/Data/spider/generate.py
'''
from Data.spider.parse import get_schemas_from_json, get_sql, Schema
CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit',
                   'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like',
             'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}
COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

# SQL_KEYWORDS = set(CLAUSE_KEYWORDS) | set(JOIN_KEYWORDS) | set(
#     WHERE_OPS) | set(UNIT_OPS) | set(AGG_OPS) | set(COND_OPS) | set(
#         SQL_OPS) | set(ORDER_OPS)
SQL_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit',
                'intersect', 'union', 'except', 'join', 'on', 'as', 'not',
                'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is',
                'exists', 'none', '-', '+', "*", '/', 'none', 'max', 'min',
                'count', 'sum', 'avg', 'and', 'or', 'intersect', 'union',
                'except', 'desc', 'asc')


def generate_col_unit(col_unit):
    agg, col, isDistinct = col_unit
    column = ''
    if isDistinct:
        column += "distinct "
    if agg != "none":
        column += agg + " ( " + col + " )"
    else:
        column += col

    return column


def generate_val_unit(val_unit):
    unit_op, col_unit1, col_unit2 = val_unit

    # none unit op
    if unit_op == "none":
        return generate_col_unit(col_unit1)
    else:
        column1 = generate_col_unit(col_unit1)
        column2 = generate_col_unit(col_unit2)
        return column1 + " " + unit_op + " " + column2


def generate_select(select):
    sel = 'select'
    if select[0]:
        sel += " distinct"
    # select column
    sel_col = []
    for col in select[1]:
        agg, val_unit = col
        if agg == "none":
            sel_col.append(generate_val_unit(val_unit))
        else:
            sel_col.append(agg + " ( " + generate_val_unit(val_unit) + " )")
    sel += " " + " , ".join(sel_col)
    return sel


def generate_cond_unit(cond_unit):
    not_op, op, val_unit, val1, val2 = cond_unit

    conds = ""
    if not_op:
        conds += "not"
    if val_unit is not None:
        conds += " " + generate_val_unit(val_unit)
        # between
        if op == "between":
            conds += " between"
            conds += " " + generate_val(val1) + " and " + generate_val(val2)
        else:
            conds += " " + op + " " + generate_val(val1)
    else:
        conds += " " + op + " " + generate_val(val1)
    return conds


def generate_val(val):
    if isinstance(val, dict):
        return "( " + generate_sql(val) + " )"
    elif isinstance(val, tuple):
        return generate_col_unit(val)
    else:
        return str(val)


def generate_from(from_clause):
    table_units = from_clause["table_units"]
    from_ = "from"
    if table_units[0][0] == "sql":
        from_ += " ( " + generate_sql(table_units[0][1]) + " )"
    else:
        from_ += " " + table_units[0][1]
        tables = set(map(lambda x: x[1], table_units))
        tables.remove(table_units[0][1])

        for index, cond_unit in enumerate(from_clause["conds"]):
            if not isinstance(cond_unit, tuple):
                continue
            condition = generate_cond_unit(cond_unit)
            first, _, second = condition.split()
            first = first.split(".")[0]
            second = second.split(".")[0]

            if first not in tables and second not in tables and first != second:
                from_ += " " + from_clause["conds"][index -
                                                    1] + " " + condition
            else:
                if first in tables:
                    from_ += " join " + first + " on"
                    tables.remove(first)
                else:
                    from_ += " join " + second + " on"
                    if second in tables:
                        tables.remove(second)
                from_ += " " + condition

    return from_


def generate_where(where):
    if not where:
        return ""
    where_ = "where"
    index = 0
    length = len(where)
    while index < length:
        where_ += " " + generate_cond_unit(where[index])
        index += 1
        if index < length:
            where_ += " " + where[index]
            index += 1
    return where_


def generate_group(groupBy):
    if not groupBy:
        return ""
    groupby = ""
    groupby += "group by"

    groups = []
    for col_unit in groupBy:
        groups.append(generate_col_unit(col_unit))
    groupby += " " + " , ".join(groups)
    return groupby


def generate_order(orderBy):
    if not orderBy:
        return ""
    orderby = ""
    orderby += "order by"

    orders = []
    for val_unit in orderBy[1]:
        orders.append(generate_val_unit(val_unit))
    orderby += " " + " , ".join(orders)
    orderby += " " + orderBy[0]

    return orderby


def generate_having(having):
    if not having:
        return ""
    having_ = ""
    having_ += " having"
    index = 0
    length = len(having)
    while index < length:
        having_ += " " + generate_cond_unit(having[index])
        index += 1
        if index < length:
            having_ += " " + having[index]
            index += 1
    return having_


def generate_sql(parsed_sql):
    sql = ''

    # select clause
    sql += generate_select(parsed_sql["select"])

    # from clause
    sql += " " + generate_from(parsed_sql["from"])

    # where clause
    sql += " " + generate_where(parsed_sql["where"])

    # group by clause
    sql += " " + generate_group(parsed_sql["groupBy"])

    # having clause
    sql += " " + generate_having(parsed_sql["having"])

    # order by clause
    sql += " " + generate_order(parsed_sql["orderBy"])

    # limit clause
    if parsed_sql["limit"] is not None:
        sql += " limit " + str(parsed_sql["limit"])

    # intersect clause
    if parsed_sql["intersect"] is not None:
        sql += " intersect " + generate_sql(parsed_sql["intersect"])

    # except clause
    if parsed_sql["except"] is not None:
        sql += " except " + generate_sql(parsed_sql["except"])

    # union clause
    if parsed_sql["union"] is not None:
        sql += " union " + generate_sql(parsed_sql["union"])

    return sql


def get_normalized_sql(sql, db_id, schemas, tables):
    schema = schemas[db_id]
    table = tables[db_id]
    schema = Schema(schema, table)
    sql_label = get_sql(schema, sql)
    new_sql = generate_sql(sql_label)

    return new_sql


if __name__ == "__main__":
    sql = "SELECT count(*) FROM professor AS T1 JOIN department AS T2 ON T1.dept_code  =  T2.dept_code WHERE DEPT_NAME  =  \"Accounting\""
    table_file = "../../Dataset/spider/tables.json"
    db_id = "college_1"

    schemas, db_names, tables = get_schemas_from_json(table_file)
    schema = schemas[db_id]
    table = tables[db_id]
    schema = Schema(schema, table)
    sql_label = get_sql(schema, sql)
    # print(sql_label)
    # for item in sql_label["from"]["conds"]:
    #     print(item)
    new_sql = generate_sql(sql_label)
    print(sql)
    print()
    print(new_sql)
    # print(sql_label)
