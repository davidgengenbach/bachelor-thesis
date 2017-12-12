import sklearn
import networkx as nx

class RelabelGraphsTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, lookup = None):
        self.lookup = lookup

    def fit(self, X, y=None, **fit_params):
        # TODO: find low-frequency node labels, merge them with other labels
        return self

    def transform(self, X, y=None, **fit_params):
        assert len(X)
        is_nx_graph = isinstance(X[0], nx.Graph)

        out = []

        if is_nx_graph:
            for graph in X:
                nodes = set(graph.nodes())
                rename_dict = {node: str(self.lookup.get(str(node), str(node))).strip() for node in nodes}
                graph = nx.relabel_nodes(graph, rename_dict)
                out.append(graph)
        else:
            for idx, (adj, nodes) in enumerate(X):
                relabeled_nodes = [str(self.lookup.get(label, label)).strip() for label in nodes]
                out.append((adj, relabeled_nodes))
        return out

