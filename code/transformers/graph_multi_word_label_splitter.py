import sklearn
import collections
import nltk
from nltk.corpus import stopwords
import networkx as nx
from utils import graph_helper
from itertools import chain


class GraphMultiWordLabelSplitter(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, remove_old_composite_labels=True, remove_stopwords=True, add_self_links=False, copy=True, lemmatizer_or_stemmer=None):
        self.remove_old_composite_labels = remove_old_composite_labels
        self.copy = copy
        self.remove_stopwords = remove_stopwords
        self.add_self_links = add_self_links
        self.lemmatizer_or_stemmer = lemmatizer_or_stemmer

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        assert len(X)
        assert isinstance(X[0], nx.Graph)

        stopwords_ = self.get_stopwords()

        X_ = [g.copy() for g in X] if self.copy else X

        all_labels = graph_helper.get_all_node_labels(X)
        if self.lemmatizer_or_stemmer:
            fn = self.lemmatizer_or_stemmer.lemmatize if hasattr(self.lemmatizer_or_stemmer, 'lemmatize') else self.lemmatizer_or_stemmer.stem
            all_labels_splitted = list(chain.from_iterable([str(x).split() for x in all_labels]))
            lookup = {k: fn(k) for k in all_labels_splitted}
            lookup = {k: v for k, v in lookup.items() if k != v}
        else:
            lookup = {}

        for graph in X_:
            nodes = sorted(graph.nodes())
            nodes_split = [(x, str(x).split()) for x in nodes]
            nodes_split = [(x, x_split) for x, x_split in nodes_split if len(x_split) > 1]
            mapping = {x: x_split for x, x_split in nodes_split}
            edges = graph.edges(data=True)
            edges_for_nodes = collections.defaultdict(lambda: [])
            for source, target, data in edges:
                edges_for_nodes[source].append((target, data, True))
                edges_for_nodes[target].append((source, data, False))
            new_edges = []
            for original, split in nodes_split:
                node_edges = edges_for_nodes[original]

                if self.remove_stopwords:
                    split = set(split) - stopwords_

                for word in split:
                    if self.add_self_links:
                        for word2 in split:
                            if word == word2: continue
                            new_edges.append((lookup.get(word, word), lookup.get(word2, word2), dict(name='ADDED')))

                    for target, data, direction in node_edges:
                        source_ = word if direction else target
                        target_ = target if direction else word
                        candidates_source = mapping.get(source_, [source_])
                        candidates_target = mapping.get(target_, [target_])
                        for s in candidates_source:
                            s = lookup.get(s, s)
                            for t in candidates_target:
                                t = lookup.get(t, t)
                                if s in stopwords_ or t in stopwords_: continue
                                new_edges.append((s, t, data))

            graph.add_edges_from(new_edges)
            if self.remove_old_composite_labels:
                graph.remove_nodes_from([x for x, x_split in nodes_split])
        return X_

    def get_stopwords(self):
        return set(stopwords.words('english')) | set([',', 'one', 'two', '-']) if self.remove_stopwords else set()
