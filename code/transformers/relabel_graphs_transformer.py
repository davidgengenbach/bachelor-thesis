import sklearn
import networkx as nx
import pickle
from utils import graph_helper
from itertools import chain
import collections

class RelabelGraphsTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, lookup_file = None, max_occurrence=2):
        self.lookup_file = lookup_file
        self.max_occurrence = max_occurrence

    def fit(self, X, y=None, **fit_params):
        assert isinstance(self.lookup_file, str)
        self.lookup = get_lookup_from_file(self.lookup_file, X, self.max_occurrence)
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


def get_lookup_from_file(label_lookup_file: str, X, max_occurrence: int=2):
    with open(label_lookup_file, 'rb') as f:
        label_lookup = pickle.load(f)

    X = graph_helper.get_graphs_only(X)

    # Get label to be renamed
    node_labels = list(chain.from_iterable([x.nodes() for x in X]))
    unique_labels = set(node_labels)
    counter = collections.Counter(node_labels)

    node_labels_to_be_renamed = set([label for label, occurrences in counter.items() if occurrences <= max_occurrence])

    lookup_ = {
        label: new_label for label, new_label in label_lookup.items() if label in node_labels_to_be_renamed
    }

    new_labels = set(lookup_.values())
    lookup__ = collections.defaultdict(list)

    for label, new_label in label_lookup.items():
        if new_label in new_labels:
            lookup__[label].append(new_label)

    lookup_ = dict(lookup_, **lookup__)
    return lookup_
