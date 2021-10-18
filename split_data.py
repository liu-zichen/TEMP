import networkx as nx
import argparse
import codecs
import random


class TaxStruct(nx.DiGraph):
    def __init__(self, edges):
        super().__init__(edges)
        self.check_useless_edge()
        self._root = ""
        for node in self.nodes:
            if self.in_degree(node) == 0:
                self._root = node
                break
        assert self._root != ""

    def check_useless_edge(self):
        """
        delete useless edge for taxonomy
        Example:
            input taxonmy:
                a->b->c
                 ————↗
            output taxonomy (deleted edge: a->c):
                a->b->c
        """
        bad_edges = []
        for node in self.nodes:
            if len(self.pred[node]) <= 1:
                continue
            # if self.out_degree(node) == 0:
            # print(node)
            for pre in self.predecessors(node):
                for ppre in self.predecessors(node):
                    if ppre != pre:
                        if nx.has_path(self, pre, ppre):
                            bad_edges.append((pre, node))
                            # print(node, pre, ppre)
        self.remove_edges_from(bad_edges)

    def all_leaf_nodes(self):
        return [node for node in self.nodes.keys() if self.out_degree(node) == 0 and self.in_degree(node) == 1]

    # sibling leaf nodes >
    def all_sibling_nodes(self, sibling_num):
        res = []
        for node in self.nodes.keys():
            if self.out_degree(node) > sibling_num:
                not_l_nodes = [s for s in self.successors(node) if self.out_degree(s) != 0 or self.in_degree(s) != 1]
                if len(not_l_nodes) <= sibling_num:
                    res.append(node)
        return res

    def all_big_nodes(self, sibling_num):
        res = []
        for node in self.nodes.keys():
            if self.out_degree(node) > sibling_num:
                res.append(node)
        return res


def spilt_data(args):
    with codecs.open(args.taxo_path, encoding='utf-8') as f:
        # TAXONOMY FILE FORMAT: relation_id <TAB> term <TAB> hypernym
        tax_lines = f.readlines()
    tax_pairs = [[w for w in line.strip().split("\t")[1:]] for line in tax_lines]
    tax_pairs = [(p[1], p[0]) for p in tax_pairs]
    tax = TaxStruct(tax_pairs)
    leaf_nodes = tax.all_leaf_nodes()
    random.seed(0)
    eval_terms = random.sample(leaf_nodes, int(len(tax.nodes) * 0.2))
    eval_lines = [list(tax.predecessors(term))[0] + "\n" for term in eval_terms]
    train_lines = ["\t".join(pair) + "\n" for pair in tax_pairs if pair[1] not in eval_terms]
    with codecs.open(args.terms_path, mode='w+', encoding='utf-8') as f:
        f.writelines([term + "\n" for term in eval_terms])
    with codecs.open(args.eval_path, mode='w+', encoding='utf-8') as f:
        f.writelines(eval_lines)
    with codecs.open(args.train_path, mode='w+', encoding='utf-8') as f:
        f.writelines(train_lines)


def split_data_by_sibling(args):
    with codecs.open(args.taxo_path, encoding='utf-8') as f:
        # TAXONOMY FILE FORMAT: relation_id <TAB> term <TAB> hypernym
        tax_lines = f.readlines()
    tax_pairs = [[w for w in line.strip().split("\t")[1:]] for line in tax_lines]
    tax_pairs = [(p[1], p[0]) for p in tax_pairs]
    eval_terms = []
    tax = TaxStruct(tax_pairs)
    all_1_nodes = tax.all_sibling_nodes(sibling_num=args.sibling)
    # random.seed(args.seed)
    eval_nums = int(len(tax.nodes) * 0.2)
    random.shuffle(all_1_nodes)
    for one_node in all_1_nodes:
        successors = list(tax.successors(one_node))
        leaf_successors = [s for s in successors if tax.out_degree(s) == 0 and tax.in_degree(s) == 1]
        random.shuffle(leaf_successors)
        eval_terms.extend(leaf_successors[0:min(len(successors) - args.sibling, len(leaf_successors))])
        if len(eval_terms) >= eval_nums:
            break
    if len(eval_terms) < eval_nums:
        print("not enough terms")
        print(len(eval_terms), eval_nums)
    with codecs.open(args.terms_path, mode='w+', encoding='utf-8') as f:
        f.writelines([term + "\n" for term in eval_terms])
    with codecs.open(args.eval_path, mode='w+', encoding='utf-8') as f:
        f.writelines([list(tax.predecessors(term))[0] + "\n" for term in eval_terms])
    with codecs.open(args.train_path, mode='w+', encoding='utf-8') as f:
        f.writelines(["\t".join(pair) + "\n" for pair in tax_pairs if pair[1] not in eval_terms])


def split_data_by_big(args):
    with codecs.open(args.taxo_path, encoding='utf-8') as f:
        # TAXONOMY FILE FORMAT: relation_id <TAB> term <TAB> hypernym
        tax_lines = f.readlines()
    tax_pairs = [[w for w in line.strip().split("\t")[1:]] for line in tax_lines]
    tax_pairs = [(p[1], p[0]) for p in tax_pairs]
    eval_terms = []
    tax = TaxStruct(tax_pairs)
    all_1_nodes = tax.all_big_nodes(sibling_num=args.sibling)
    # random.seed(args.seed)
    eval_nums = int(len(tax.nodes) * 0.2)
    random.shuffle(all_1_nodes)
    for one_node in all_1_nodes:
        successors = list(tax.successors(one_node))
        leaf_successors = [s for s in successors if tax.out_degree(s) == 0 and tax.in_degree(s) == 1]
        random.shuffle(leaf_successors)
        eval_terms.extend(leaf_successors[0:min(len(successors) - args.sibling, len(leaf_successors))])
    if len(eval_terms) >= eval_nums:
        eval_terms = random.sample(eval_terms, eval_nums)
    if len(eval_terms) < eval_nums:
        print("not enough terms")
        print(len(eval_terms), eval_nums)
    with codecs.open(args.terms_path, mode='w+', encoding='utf-8') as f:
        f.writelines([term + "\n" for term in eval_terms])
    with codecs.open(args.eval_path, mode='w+', encoding='utf-8') as f:
        f.writelines([list(tax.predecessors(term))[0] + "\n" for term in eval_terms])
    with codecs.open(args.train_path, mode='w+', encoding='utf-8') as f:
        f.writelines(["\t".join(pair) + "\n" for pair in tax_pairs if pair[1] not in eval_terms])
