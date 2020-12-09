'''
Author: your name
Date: 1970-01-01 01:00:00
LastEditTime: 2020-08-24 17:59:47
LastEditors: Please set LastEditors
Description: some tools
FilePath: /Tree2Seq/Utils/tools.py
'''
import datetime
import numpy as np


def transform_time(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=6)
    return beijing_time.timetuple()


def cut(data, pad_index):
    flag = data != pad_index
    max_len = flag.sum(dim=1).max()
    return data[:, :max_len]


def get_position_relations(node_num: int, k: int) -> np.ndarray:
    position = np.zeros((node_num, node_num), dtype=np.int)
    idx = np.arange(node_num)
    for i in range(node_num):
        position[i] = idx - i

    position = position + k
    position[position < 0] = 0
    position[position > 2 * k] = 2 * k

    return position
