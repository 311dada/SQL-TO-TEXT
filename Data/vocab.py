'''
@Author: your name
@Date: 2020-07-24 11:07:55
LastEditTime: 2020-09-04 22:14:20
LastEditors: Please set LastEditors
@Description: the vocabulary class inherited from torchtext vocabulary
@FilePath: /Graph2Seq_v2/Data/vocab.py
'''
import torchtext
from typing import List, Tuple, Union, Dict
import json
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


class Vocabulary(torchtext.vocab.Vocab):
    """A vocabulary based on torchtext vocabulary.
    """
    def __init__(self,
                 counter,
                 max_size=None,
                 min_freq=1,
                 specials=['<unk>', '<pad>'],
                 vectors=None,
                 unk_init=None,
                 vectors_cache=None,
                 specials_first=True):
        super(Vocabulary, self).__init__(counter,
                                         max_size=max_size,
                                         min_freq=min_freq,
                                         specials=specials,
                                         vectors=vectors,
                                         unk_init=unk_init,
                                         vectors_cache=vectors_cache,
                                         specials_first=specials_first)

    def get_idx(self, tok: str, tok2idx_map=None) -> int:
        """Return the index of tok.

        Args:
            tok (str): token

        Returns:
            int: index
        """
        if tok in self.stoi:
            return self.stoi[tok]
        elif tok2idx_map is not None:
            if tok in tok2idx_map:
                return tok2idx_map[tok]
            else:
                lemma = lemmatizer.lemmatize(tok)
                if lemma in tok2idx_map:
                    return tok2idx_map[lemma]
                else:
                    return self.stoi["<unk>"]
        else:
            return self.stoi["<unk>"]

    @property
    def pad_idx(self):
        return self.get_idx("<pad>")

    @property
    def size(self):
        return len(self.itos)

    @property
    def unk_idx(self):
        return self.get_idx("<unk>")

    def to_toks(self, ids, idx2tok_map=None) -> List:
        """Return the token list according to index list.

        Args:
            ids (List): index list

        Returns:
            List: token list
        """
        toks = []
        length = len(self.itos)
        for idx in ids:
            if idx < length:
                tok = self.itos[idx]
            elif idx2tok_map is not None and idx in idx2tok_map:
                tok = idx2tok_map[idx]
            else:
                tok = "<unk>"

            if tok == "<eos>":
                break
            else:
                toks.append(tok)
        return toks

    def to_ids_2_dim(
        self,
        toks,
        add_end: bool = False,
        pad_len: int = None,
        pad: bool = True,
        unk: bool = False,
        TOK2IDX_MAP_LIST=None
    ) -> Union[List[List], Tuple[List[List], List[List], List[Dict]]]:
        """Transform 2-dim toks list into ids list. This function will pad
        the toks automatically.

        Args:
            toks (List[List[str]]): toks list.
            add_end (bool): if add begin symbol and end symbol.
            pad_len (Optional[int]): the max len to pad into.

        Returns:
            List[List]: index representation
        """
        lengths: List = []

        src2trg_map_list = []
        idx2tok_map_list = []
        tok2idx_map_list = []

        for index, tok_list in enumerate(toks):
            # add start and end token
            if add_end:
                toks[index] = ["<bos>"] + tok_list + ["<eos>"]

            # record length
            lengths.append(len(toks[index]))

            # transform token to index
            cur_oov_idx = len(self)
            src2trg_map = []
            tok2idx_map = dict()
            idx2tok_map = dict()
            TOK2IDX_MAP = None
            if TOK2IDX_MAP_LIST is not None:
                TOK2IDX_MAP = TOK2IDX_MAP_LIST[index]
            for idx, tok in enumerate(toks[index]):
                toks[index][idx] = self.get_idx(tok, TOK2IDX_MAP)

                if unk:
                    if tok not in self.stoi and tok not in tok2idx_map:
                        src2trg_map.append(cur_oov_idx)
                        idx2tok_map[cur_oov_idx] = tok
                        tok2idx_map[tok] = cur_oov_idx
                        cur_oov_idx += 1
                    elif tok not in self.stoi:
                        src2trg_map.append(tok2idx_map[tok])
                    else:
                        src2trg_map.append(toks[index][idx])
            src2trg_map_list.append(src2trg_map)
            idx2tok_map_list.append(idx2tok_map)
            tok2idx_map_list.append(tok2idx_map)

        if pad:
            max_len: int = pad_len or max(lengths)
            # pad with pad index
            pad_index: int = self.stoi['<pad>']
            for index, tok_list in enumerate(toks):
                cur_pad_num = (max_len - len(tok_list))
                toks[index] += [pad_index] * cur_pad_num
                if unk:
                    src2trg_map_list[index] += [pad_index] * cur_pad_num

        if not unk:
            return toks
        else:
            return toks, src2trg_map_list, idx2tok_map_list, tok2idx_map_list

    def to_ids_3_dim(self,
                     toks,
                     pad_schema_len: int = None,
                     pad_tok_len: int = None,
                     pad_schema: bool = True,
                     pad_tok: bool = True) -> List[List[List]]:
        """Transform 3-dim toks list into ids list. This function will pad
        the toks automatically.

        Args:
            toks (List[List[List[str]]]): toks list.
            pad_schema_len (Option[int]): the max schema len to pad into.
            pad_tok_len (Option[int]): the max toks number to pad into.

        Returns:
            List[List[List]]: index representation
        """
        # pad schema (tables and columns) first
        schema_lengths: List = []
        for columns in toks:
            schema_lengths.append(len(columns))

        if pad_schema:
            max_schema_len: int = pad_schema_len or max(schema_lengths)

            for schema_index, schema in enumerate(toks):
                toks[schema_index] += [[] for _ in range(max_schema_len -
                                                         len(schema))]

        # replace token to index and record column length
        column_lengths: List[List] = [[] for _ in range(len(toks))]
        max_col_len: int = 0
        for schema_index, schema in enumerate(toks):
            for column_index, column in enumerate(schema):
                for tok_index, tok in enumerate(column):
                    toks[schema_index][column_index][tok_index] = self.get_idx(
                        tok)
                column_lengths[schema_index].append(len(column))
                max_col_len = max(max_col_len, len(column))

        if pad_tok:
            pad_index: int = self.stoi["<pad>"]

            if pad_tok_len is not None:
                max_col_len = pad_tok_len
            # pad columns
            for schema_index, schema in enumerate(toks):
                for column_index, column in enumerate(schema):
                    toks[schema_index][column_index] += [pad_index] * (
                        max_col_len - len(column))

        return toks

    def add_words(self, words):
        for word in words:
            if word not in self.stoi:
                self.stoi[word] = len(self.stoi)
                self.itos.append(word)

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump([self.itos, self.stoi], f)

    def load(self, filename):
        with open(filename, 'r') as f:
            self.itos, self.stoi = json.load(f)
