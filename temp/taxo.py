import networkx as nx


class TaxStruct(nx.DiGraph):
    def __init__(self, edges):
        super().__init__(edges)
        self.check_useless_edge()
        self._root = ""
        for node in self.nodes:  # find root
            if self.in_degree(node) == 0:
                self._root = node
                break
        assert self._root != ""
        self._node2path = dict()
        for node in self.nodes.keys():
            self._node2path[node] = list(reversed(nx.shortest_path(self, source=self._root, target=node)))
        self.leaf_nodes = self.all_leaf_nodes()

    def check_useless_edge(self):
        """
        delete useless edges
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
        # 根据是否只要单一父节点的叶节点可进行更改
        return [node for node in self.nodes.keys() if self.out_degree(node) == 0 and self.in_degree(node) == 1]

    def get_node2full_path(self):
        node2full_path = {}
        for node in self.nodes:
            paths = nx.all_simple_paths(self, source=self.root, target=node)
            all_nodes = set([node for path in paths for node in path])
            node2full_path[node] = all_nodes
        return node2full_path

    @property
    def node2path(self):
        return self._node2path

    @property
    def root(self):
        return self._root
