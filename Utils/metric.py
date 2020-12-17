'''
@Author: your name
@Date: 2020-07-25 02:20:20
LastEditTime: 2020-09-05 17:13:12
LastEditors: Please set LastEditors
@Description: metric calculation module
@FilePath: /Graph2Seq_v2/Utils/metric.py
'''
from typing import List, Union
from Data.vocab import Vocabulary
from sacrebleu import corpus_bleu


def join_toks(toks: List[str], to_join=False) -> Union[str, List[str]]:
    """Join tokens to a sentence.

    Args:
        toks (List[str]): tokens list.

    Returns:
        str: joined sentence.
        List[str]: normalized tokens.
    """
    new_toks = []
    index = 0
    length = len(toks)

    while index < length:
        tok = toks[index]
        if index < length - 1:
            next_tok = toks[index + 1]
            if next_tok == "'s":
                if tok[-1] == "s":
                    cur_tok = tok + "'"
                else:
                    cur_tok = tok + "'s"
                index += 2
                new_toks.append(cur_tok)
                continue
            elif next_tok == "'":
                cur_tok = tok + "'"
                index += 2
                new_toks.append(cur_tok)
                continue
        index += 1
        new_toks.append(tok)

    if to_join:
        return " ".join(new_toks)
    return new_toks


def combine_same_ref(preds_, refs_):
    preds = []
    refs = []

    pre = ""

    for index, pred in enumerate(preds_):
        if pred == pre:
            refs[-1].append(refs_[index])
        else:
            preds.append(pred)
            pre = pred
            refs.append([refs_[index]])

    ref1 = []
    ref2 = []

    for ref in refs:
        ref1.append(ref[0])
        if len(ref) > 1:
            ref2.append(ref[1])
        else:
            ref2.append(ref[0])

    return preds, [ref1, ref2]


def get_metric(preds: List[List],
               labels: List,
               vocab: Vocabulary,
               delexicalize=False,
               val_map_list=None,
               idx2tok_map_list=None) -> float:
    """Calculate the score between preds and labels. Support BLEU4 only now.

    Args:
        preds (List[List]): prediction list
        labels (List): label list

    Returns:
        float: BLEU4
    """
    # transform index representation into token list
    preds = [
        vocab.to_toks(pred, idx2tok_map=idx2tok_map_list[index])
        for index, pred in enumerate(preds)
    ]

    if delexicalize:
        lexi_preds = []
        for index, pred in enumerate(preds):
            val_map = val_map_list[index]
            pred = ' '.join(pred)

            for val in val_map:
                pred = pred.replace(val_map[val], val)
            lexi_preds.append(pred.lower().split())
        preds = lexi_preds

    preds = [join_toks(pred, to_join=True) for pred in preds]

    preds_, refs_ = combine_same_ref(preds, labels)

    return corpus_bleu(preds_, refs_, force=True,
                       lowercase=True).score, preds, labels
