"""
    SOME UTILITY FUNCTIONS FOR MODEL
"""
import torch


def bin2inf(x):
    x = x.type(torch.bool)
    save = torch.zeros_like(x).type(torch.float)
    mask = torch.ones_like(x).type(torch.float) * -1e4
    return torch.where(x, save, mask)


def mean_pooling(x, mask=None):
    if mask is not None:
        x = x * mask.type(torch.float)
        return x.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
    return torch.mean(x, dim=1, keepdim=True)


def max_pooling(x, mask=None):
    if mask is not None:
        mask = bin2inf(mask)
        x = x + mask
        return torch.max(x, dim=1, keepdim=True)[0]
    return torch.max(x, dim=1, keepdim=True)


def get_bin_mask(x, pad):
    return x != pad
