import sklearn
import networkx as nx
import pickle
import os
from utils import graph_helper
from itertools import chain
import collections

class RelabelGraphsTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, dataset:str = None, threshold: str=None, topn: str=None, max_occurrence=2, remove_unseen=True):
        self.threshold = threshold
        self.dataset = dataset
        self.topn = topn
        self.max_occurrence = max_occurrence
        self.remove_unseen = remove_unseen

    def fit(self, X, y=None, **fit_params):
        self.lookup_ = get_lookup_from_file(self.dataset, self.threshold, self.topn, X, remove_unseen=self.remove_unseen, max_occurrence=self.max_occurrence)
        return self

    def transform(self, X, y=None, **fit_params):
        assert len(X)
        is_nx_graph = isinstance(X[0], nx.Graph)

        out = []
        if is_nx_graph:
            for graph in X:
                nodes = set(graph.nodes())
                rename_dict = {node: str(self.lookup_.get(str(node), str(node))).strip() for node in nodes}
                graph = nx.relabel_nodes(graph, rename_dict)
                out.append(graph)
        else:
            for idx, (adj, nodes) in enumerate(X):
                relabeled_nodes = [str(self.lookup_.get(label, label)).strip() for label in nodes]
                out.append((adj, relabeled_nodes))
        return out

def get_node_labels(X):
    return list(chain.from_iterable([x.nodes() for x in X]))

def get_node_labels_below_given_threshold(X, max_occurrence: int = 2):
    # Get label to be renamed
    node_labels = get_node_labels(X)

    unique_labels = set(node_labels)
    counter = collections.Counter(node_labels)

    node_labels_to_be_renamed = set([label for label, occurrences in counter.items() if occurrences <= max_occurrence])
    return node_labels_to_be_renamed


def get_lookup_from_file(dataset, threshold, topn, X, remove_unseen=True, max_occurrence: int=2, label_lookup_folder='data/embeddings/graph-embeddings'):
    label_lookup_file = '{}/{}.threshold-{}.topn-{}.label-lookup.npy'.format(label_lookup_folder, dataset, threshold, topn)

    if not os.path.exists(label_lookup_file):
        raise FileNotFoundError('Label lookup file not found: {}'.format(label_lookup_file))

    with open(label_lookup_file, 'rb') as f:
        label_lookup = pickle.load(f)

    node_labels = set(get_node_labels(X))
    node_labels_to_be_renamed = get_node_labels_below_given_threshold(X, max_occurrence)

    lookup_ = {
        label: new_label for label, new_label in label_lookup.items() if label in node_labels_to_be_renamed
    }

    new_labels = set(lookup_.values())
    lookup__ = collections.defaultdict(list)

    for label, new_label in label_lookup.items():
        # Only rename labels that are seen (= are node labels in the fitted graphs)
        # This is important so that no data implicitely leaks from the test to the train set
        if new_label in new_labels and (not remove_unseen or label in node_labels):
            lookup__[label].append(new_label)

    lookup_ = dict(lookup_, **lookup__)
    return lookup_
