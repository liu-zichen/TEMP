from temp.taxo import TaxStruct
from torch.utils.data import Dataset as TorchDataset
import transformers
import torch
import numpy as np
import random


class Sampler:
    def __init__(self, tax_graph: TaxStruct):
        self._tax_graph = tax_graph
        self._nodes = list(self._tax_graph.nodes.keys())

    def sampling(self):
        margins = []
        pos_paths = []
        neg_paths = []
        for node, path in self._tax_graph.node2path.items():
            if node == self._tax_graph.root:
                continue
            while True:
                neg_node = random.choice(self._nodes)
                if neg_node != path[1] and neg_node != node:
                    break
            pos_paths.append(path)
            neg_path = [node] + self._tax_graph.node2path[neg_node]
            neg_paths.append(neg_path)
            margins.append(self.margin(path, neg_path))
        return pos_paths, neg_paths, margins

    @staticmethod
    def margin(path_a, path_b):
        com = len(set(path_a).intersection(set(path_b)))
        return max(min((abs(len(path_a) - com) + abs(len(path_b) - com)) / com, 2), 0.5)


class Dataset(TorchDataset):
    def __init__(self, sampler, tokenizer, word2des, padding_max=256,
                 margin_beta=0.1):
        self._sampler = sampler
        self._word2des = word2des
        self._padding_max = padding_max
        self._margin_beta = margin_beta
        self._tokenizer = tokenizer
        if self._sampler is not None:
            self._pos_paths, self._neg_paths, self._margins = self._sampler.sampling()

    def __len__(self):
        return len(self._pos_paths)

    def __getitem__(self, item):
        pos_path = self._pos_paths[item]
        neg_path = self._neg_paths[item]
        margin = self._margins[item]
        pos_ids, pos_type_ids, pos_attn_masks = self.encode_path(pos_path)
        neg_ids, neg_type_ids, neg_attn_masks = self.encode_path(neg_path)
        return dict(pos_ids=pos_ids,
                    neg_ids=neg_ids,
                    pos_type_ids=pos_type_ids,
                    neg_type_ids=neg_type_ids,
                    pos_attn_masks=pos_attn_masks,
                    neg_attn_masks=neg_attn_masks,
                    margin=torch.FloatTensor([margin * self._margin_beta]))

    def encode_path(self, path):
        des_sent = self._word2des[path[0]][0]
        def_sent = str(" " + self._tokenizer.unk_token + " ").join(path)
        encode = self._tokenizer.encode_plus(des_sent, def_sent, add_special_tokens=True, return_token_type_ids=True
                                             # return_tensors='pt'
                                             )
        input_len = len(encode["input_ids"])
        assert input_len <= self._padding_max
        encode["input_ids"] = encode["input_ids"] + [self._tokenizer.pad_token_id] * (self._padding_max - input_len)
        encode["token_type_ids"] = encode["token_type_ids"] + [0] * (self._padding_max - input_len)
        encode["attention_mask"] = encode["attention_mask"] + [0] * (self._padding_max - input_len)
        return torch.LongTensor(encode["input_ids"]), torch.LongTensor(encode["token_type_ids"]), torch.LongTensor(
            encode["attention_mask"])
