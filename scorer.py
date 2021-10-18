def score(results, tax_graph, trues):
    def get_wu_p(node_a, node_b):
        # if node_a == node_b:
        #     return 1.0
        full_path_a = node2full_path[node_a]
        full_path_b = node2full_path[node_b]
        com = full_path_a.intersection(full_path_b)
        lca_dep = 1
        for node in com:
            lca_dep = max(len(tax_graph.node2path[node]), lca_dep)
        dep_a = len(tax_graph.node2path[node_a])
        dep_b = len(tax_graph.node2path[node_b])
        res = 2.0 * float(lca_dep) / float(dep_a + dep_b)
        # assert res <= 1
        return res

    node2full_path = tax_graph.get_node2full_path()
    wu_p, acc, mrr = 0.0, 0.0, 0.0
    wrong_set = []
    ii = 0
    for result, ground_true in zip(results, trues):
        if result[0] == ground_true:
            acc += 1
        else:
            wrong_set.append([ii, result[0], ground_true])
        num = 0
        for i, r in enumerate(result):
            if r == ground_true:
                num = i + 1.0
                break
        mrr += 1.0 / num
        wu_p += get_wu_p(result[0], ground_true)
        ii += 1
    acc /= float(len(results))
    mrr /= float(len(results))
    wu_p /= float(len(results))

    return acc, mrr, wu_p, wrong_set
