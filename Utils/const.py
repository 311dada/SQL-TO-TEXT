# DATA = "wikisql"

# NOT_VALID_TOKS = ["root", "column", "cond"]

# # ------------------- WIKISQL ------------------- #
# WIKISQL_AGG_OPS = ['', 'max', 'min', 'count', 'sum', 'avg']
# WIKISQL_COND_OPS = ['=', '>', '<', 'op']
# WIKISQL_NODE_TYPES = {
#     "root": 0,
#     "select": 1,
#     "agg": 2,
#     "and": 3,
#     "op": 4,
#     "val": 5,
#     "col": 6,
#     "column": 7,
#     "cond": 8,
#     "where": 9
# }
# WIKISQL_NODE_TYPE_NUM = len(WIKISQL_NODE_TYPES)

# ------------------- SPIDER ------------------- #
SPIDER_MONTH = [
    None, 'january', 'february', 'march', 'april', 'may', 'june', 'july',
    'august', 'september', 'october', 'november', 'december'
]

# SPIDER_NODE_TYPES = {
#     "root": 0,
#     "select": 1,
#     "agg": 2,
#     "logic": 3,
#     "op": 4,
#     "val": 5,
#     "col": 6,
#     "tab": 7,
#     "from": 8,
#     "where": 9,
#     "group by": 10,
#     "order by": 11,
#     "limit": 12,
#     "join": 13,
#     "unit_op": 14,
#     "order": 15,
#     "set": 16,
#     "distinct": 17,
#     "having": 18,
#     "column": 19,
#     "cond": 20,
#     "bracket": 21,
#     "unit": 22,
#     "table": 23,
#     "val_unit": 24,
#     "comma": 25,
# }

# SPIDER_NODE_TYPE_NUM = len(SPIDER_NODE_TYPES)

SPIDER_MAP = {
    "los angeles": "la",
    "eggs": "egg",
    "usa": "us",
    "billy cobham": "billy cobam",
    "lucas": "luca",
    "balls to the wall": "ball to the wall",
    "mpeg audio file": "mpeg",
    "daan": "dean",
    "activitor": "activator",
    "collectible card game": "collectible cards",
    "italy": "italian",
    "united states": "us",
    "canada": "canadian",
    "asstprof": "assistant professors",
    "completed": "complete",
    "a puzzling parallax": "a puzzling pattern",
    "herbs": "herb",
    "hiram , georgia": "hiram, goergia",
    "protoporphyrinogen ix": "heme",
    "tokyo , japan": "tokyo,japan",
    "annaual meeting": "annual meeting",
    "san jose state university": "san jose state",
    "brittany harris": "britanny harris",
    "mortgages": "mortgage",
    "accounting": "accoutning",
    "ph.d.": "pd.d",
    "new policy application": "upgrade a policy",
    "sweazy": "sweaz",
    "2010-10-23": "oct 23, 2010",
    "2010-09-21": "sep 21, 2010",
    "2007-11-05": "november 5th, 2007",
    "2009-07-05": "july 5th, 2009",
    "2002-06-21": "june 21, 2002",
    "1987-09-07": "september 7th, 1987",
    "region0": "bay area",
    "director_name0": "kevin spacey",
    "actor_name0": "kevin spacey",
    # "category_category_name0":
    # ["vintner grill", "flat top grill", "mgm grand buffet"],
    "uk": "british",
    "united kingdom": "uk",
    "new york city": "new york",
    "division 1": "1",
    "city mall": "mall",
    "researcher": "research",
    "london gatwick": "gatwick",
    "la": "louisiana",
    "rylan": "ryan",
    "nominated": "nomination",
    "f": "girl",
    "sesame": "cumin",
    "goalie": "goal",
    "international": "interanation"
}

UP_TYPES = {
    "table": 0,
    "column": 1,
    "clause": 2,
    "set": 3,
    "unit": 4,
    "condition": 5,
    "root": 6,
    "logic": 7
}

DOWN_TYPES = {"distinct": 0, "agg": 1, "token": 2, "order": 3, "not": 4}

SCHEMA_RELATIONS = {
    "SAME-TABLE": 0,
    "FOREIGN-KEY-COL-F": 1,
    "FOREIGN-KEY-COL-R": 2,
    "PRIMARY-KEY-F": 3,
    "BELONGS-TO-F": 4,
    "PRIMARY-KEY-R": 5,
    "BELONGS-TO-R": 6,
    "FOREIGN-KEY-TAB-F": 7,
    "FOREIGN-KEY-TAB-R": 8,
    "FOREIGN-KEY-TAB-B": 9,
    "NONE": 10,
}

UP_TYPE_NUM = len(UP_TYPES)
DOWN_TYPE_NUM = len(DOWN_TYPES)
UP_SCHEMA_NUM = len(SCHEMA_RELATIONS)

RGT_VOCAB_PATH = "Cache/vocab/RGT"
RGT_MODEL_PATH = "Checkpoints/RGT"

# RELATIVE_VOCAB_PATH = "Cache/vocab/RelativeTransformer"
RELATIVE_MODEL_PATH = "Checkpoints/RelativeTransformer"

# TRANSFORMER_VOCAB_PATH = "Cache/vocab/Transformer"
TRANSFORMER_MODEL_PATH = "Checkpoints/Transformer"

SEQ_VOCAB_PATH = "Cache/vocab/Seq"
SEQ_MODEL_PATH = "Checkpoints/Seq"


