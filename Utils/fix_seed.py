'''
@Author: your name
@Date: 2SEED2SEED-SEED7-25 SEED3:17:23
LastEditTime: 2020-09-06 21:13:08
LastEditors: Please set LastEditors
@Description: fix seed
@FilePath: /Graph2Seq_v2/Utils/fix_seed.py
'''
import random
import numpy as np
import torch


def fix_seed(SEED: int) -> None:
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
