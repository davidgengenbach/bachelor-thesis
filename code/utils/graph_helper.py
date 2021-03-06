from utils.constants import *

import typing
from glob import glob
import collections
import os

import networkx as nx
from joblib import Parallel, delayed
from scipy.sparse import lil_matrix
import numpy as np

from preprocessing import preprocessing
from utils import cooccurrence, filename_utils, dataset_helper


def get_graph_type_from_filename(x):
    for t in GRAPH_TYPES:
        if t in x:
            return t
    return None


def add_shortest_path_edges(graph, cutoff=2):
    if graph.number_of_edges() == 0 or graph.number_of_nodes() == 0:
        return
    shortest_paths = nx.all_pairs_shortest_path(graph, cutoff=cutoff)
    for source, target_dict in shortest_paths.items():
        for target, path in target_dict.items():
            graph.add_edge(source, target, attr_dict={'weight': 1 / len(path)})


def convert_dataset_to_co_occurence_graph_dataset(X, n_jobs=4, **cooccurrence_kwargs):
    mats = Parallel(n_jobs=n_jobs)(delayed(cooccurrence.get_coocurrence_matrix)(text, **cooccurrence_kwargs) for text in X)
    graphs = Parallel(n_jobs=n_jobs)(delayed(convert_from_numpy_to_nx)(*mat) for mat in mats)
    return graphs


def get_all_node_labels(graphs):
    labels = []
    for graph in graphs:
        assert isinstance(graph, nx.Graph)
        labels += graph.nodes()
    return labels


def get_all_node_labels_uniq(graphs, as_sorted_list=True):
    """Returns all unique labels in a list of graphs.

    Args:
        graphs (list(nx.Graph)): list of graphs

    Returns:
        list(str): the unique labels
    """
    labels = set()
    for graph in graphs:
        if isinstance(graph, tuple) or isinstance(graph, np.ndarray):
            nodes = graph[1]
        elif isinstance(graph, nx.Graph):
            nodes = graph.nodes()
        else:
            assert False
        labels |= set(nodes)
    return sorted(list([str(x) for x in labels])) if as_sorted_list else labels


def convert_from_numpy_to_nx(word2id, id2word, mat):
    """Converts from the co-occurrence format to the networkx

    Args:
        word2id (dict): keys are the labels, values are the ids
        id2word (dict): vice versa
        mat (sparse matrix): the adjacency matrix

    Returns:
        networkx.Graph: the graph corresponding to the adjacency matrix
    """
    graph = nx.from_numpy_matrix(mat.toarray())
    nx.relabel_nodes(graph, mapping=id2word, copy=False)
    return graph


def get_graphs_from_folder(folder, ext='gml', undirected=False, verbose=False):
    """Reads in and parses all gml graphs from a folder.

    Args:
        folder (str): where to search for graphs
        ext (str, optional): the file extension
        undirected (bool, optional): whether to keep or remove edge directions
        verbose (bool, optional): log more or less

    Returns:
        tuple of lists: first list contains the graphs, second the corresponding labels
    """
    X, Y = [], []
    empty_graphs = []

    def get_all_files(folder):
        files = glob('{}/**/*.{}'.format(folder, ext), recursive=True) + glob('{}/**/*.graph'.format(folder), recursive=True)
        return sorted(files)

    def extract_y_and_id(file_path):
        filename = file_path.rsplit('/', 1)[1]

        if '.graph' in filename:
            # Like: data/graphs/ling-spam-single_v2/all/no_spam_0000/fullgraph.graph
            t = file_path.rsplit('/', 2)[1]
        else:
            # Like: data/graphs/ling-spam-single_v1/no_spam_0000.gml
            t = filename.rsplit('.', 1)[0]

        return t.rsplit('_', 1)

    files = get_all_files(folder)

    for idx, graph_file in enumerate(files):
        topic, graph_id = extract_y_and_id(graph_file)

        topic_and_id = '{:20} - {:5}'.format(topic, graph_id)

        with open(graph_file) as f:
            graph_str = f.read()
        graph = get_gml_graph(graph_str, undirected)
        # Ignore empty graphs and graphs that could not be parsed
        if graph and graph.number_of_nodes() > 0 and graph.number_of_edges() > 0:
            X.append((graph, graph_id))
            Y.append(topic)
        else:
            if verbose:
                print("Empty graph: {}".format(topic_and_id))
            empty_graphs.append((topic_and_id, graph_file))

    print('Empty graphs found: {}'.format(len(empty_graphs)))
    assert len(X) and len(Y), 'X or Y empty'
    assert len(X) == len(Y), 'X has not the same dimensions as Y'
    return X, Y


def convert_adjs_tuples_to_graphs(X, copy=False):
    if not isinstance(X[0], tuple):
        return

    if copy:
        X_ = []
    else:
        X_ = X

    for idx, (adj, labels) in enumerate(X):
        g = nx.from_scipy_sparse_matrix(adj)
        nx.relabel_nodes(g, {idx: label for idx, label in enumerate(labels)}, copy=False)
        if copy:
            X_.append(g)
        else:
            X[idx] = g
    return X_


def convert_graphs_to_adjs_tuples(X, copy=False) -> typing.Iterable:
    """Converts the graphs from the nx.Graph format to a tuple.
    Note: this function changes the input!

    Args:
        X (list(nx.graph)): the graphs

    Returns:
        list(tuple): a list of tuples where the first tuple element is an adjacency matrix and the second a list of labels
    """
    assert len(X)

    if (isinstance(X[0], tuple) or isinstance(X[0], np.ndarray)) and not isinstance(X[0][0], nx.Graph):
        return X

    X_ = [] if copy else X

    for idx, graph in enumerate(X):
        if (isinstance(graph, np.ndarray) or isinstance(graph, tuple)) and isinstance(graph[0], nx.Graph):
            graph = graph[0]

        nodes = sorted(graph.nodes())
        if len(nodes) == 0 or nx.number_of_edges(graph) == 0:
            out = (lil_matrix(1, 1), ['no_label'])
        else:
            out = (nx.adjacency_matrix(graph, nodelist=nodes), nodes)

        if copy:
            X_.append(out)
        else:
            X[idx] = out
    return X_


def get_gml_graph(graph_str, undirected=False, num_tries=20, verbose=False):
    """Given a gml string, this function returns a networkx graph.
    Mostly tries to resolve "duplicate node label" exceptions by replacing node labels with the first occurrence of that label.

    Args:
        graph_str (str): the graph in gml
        undirected (bool, optional): Whether to keep the direction
        num_tries (int, optional): how often to try to solve "duplicate node label" exceptions
        verbose (bool, optional): logs more

    Returns:
        networkx.(Di)Graph or None: Returns a networkx graph on success, None otherwise
    """
    graph_str_lines = graph_str.split('\n')
    for idx, line in enumerate(graph_str_lines):
        if line.startswith('label '):
            next_line = graph_str_lines[idx + 1]
            label = next_line.replace('name', 'label', 1)
            graph_str_lines[idx] = label

    def convert_to_nx(graph_):
        try:
            graph = nx.parse_gml('\n'.join(graph_str_lines))
            if undirected:
                graph = graph.to_undirected()
            return graph
        except nx.NetworkXError as e:
            return e

    for i in range(num_tries):
        result = convert_to_nx('\n'.join(graph_str_lines))
        if isinstance(result, nx.NetworkXError):
            is_duplicate_label_error = str(result).startswith('node label ') and str(result).endswith(' is duplicated')
            if not is_duplicate_label_error:
                break
            # This code resolves the duplicate label error by replacing the
            # occurrences of a duplicate label by its first occurrence
            delim = "'" if str(result).count('\'') == 2 else '"'
            duplicate_label = "'".join(str(result).split(delim)[1:-1])
            if verbose:
                print('\tFound duplicate node label: "{}", trying to replace with first occurrence'.format(duplicate_label))
            # Try to recover
            occurences = []
            for idx, line in enumerate(graph_str_lines):
                if line.startswith('label '):
                    label = line.replace('label', '', 1).strip()[1:-1]
                    label_id = graph_str_lines[idx - 1].replace('id ', '').strip()
                    if label == duplicate_label.strip():
                        start = idx - 2
                        end = idx + 2
                        occurences.append((start, end, label_id))
            assert len(occurences) > 1
            first_occurence_id = occurences[0][2]
            for start, end, label_id in occurences[1:]:
                for i in range(end - start + 1):
                    graph_str_lines[start + i] = ''
                for idx, line in enumerate(graph_str_lines):
                    graph_str_lines[idx] = line.replace('source {}'.format(label_id), 'source {}'.format(
                        first_occurence_id)).replace('target {}'.format(label_id), 'target {}'.format(first_occurence_id))
        else:
            return result
    return None


def _parse_graph(graph_definition: str):
    modes = [[], [], []]
    mode = -1
    for line in graph_definition.splitlines():
        line = line.strip()
        if line == '':
            continue
        if line.startswith('#'):
            mode += 1
            continue
        modes[mode].append(line)
    vertices, adj_rows, clazz = modes

    num_vertices = len(vertices)
    adj_matrix = lil_matrix((num_vertices, num_vertices), dtype=np.uint16)

    for row_idx, adj_row in enumerate(adj_rows):
        parts = [int(x) - 1 for x in adj_row.split(',')]
        adj_matrix[row_idx, parts] = 1

    return adj_matrix, vertices, clazz[0]


def get_combined_text_graph_dataset(graph_cache_file, use_ana=False) -> typing.Tuple[typing.List[typing.Tuple], typing.List]:
    dataset_name = filename_utils.get_dataset_from_filename(graph_cache_file)

    X_text, Y_text = dataset_helper.get_dataset(dataset_name + ('-ana' if use_ana else ''))
    X_graph, Y_graph = dataset_helper.get_dataset_cached(graph_cache_file)

    # Same length but has ID
    if len(X_graph) == len(X_text) and (not isinstance(X_graph[0], tuple) or not isinstance(X_graph[0][1], str)):
        return list(zip(X_graph, X_text, [None] * len(X_graph))), Y_graph

    # Get class to class ids mapping
    class_2_id = collections.defaultdict(lambda: [])
    for x, y in zip(X_text, Y_text):
        class_2_id[y].append(x)

    X_combined, Y_combined = [], Y_graph
    for (x_graph, y_id), y_graph in zip(X_graph, Y_graph):
        y_id = int(y_id)
        X_combined.append((x_graph, class_2_id[y_graph][y_id], y_id))

    return X_combined, Y_combined


def get_adjs_only(X, copy=True):
    X_ = [] if copy else X

    if not len(X) or not isinstance(X[0], tuple):
        return np.copy(X)

    for idx, x in enumerate(X):
        adj = x[0]
        if copy:
            X_.append(adj)
        else:
            X[idx] = adj

    return X_


def get_graphs_with_mutag_enzyme_format(folder, as_adj_tuple: bool=False):
    graphs = glob('{}/*.graph'.format(folder))
    X, Y = [], []
    for graph_file in graphs:
        with open(graph_file) as f:
            adj_matrix, vertices, clazz = _parse_graph(f.read())
        X.append((adj_matrix, vertices))
        Y.append(clazz)
    if not as_adj_tuple:
        X = convert_adjs_tuples_to_graphs(X)
    return X, Y


def graph_to_text(graph, use_edges=True):
    text = []
    for source, target, data in graph.edges(data=True):
        if use_edges and 'name' in data:
            t = [source, data['name'], target]
        else:
            t = [source, target]
        text.append(' '.join([str(x) for x in t]))
    return '. '.join(text)


def warmup_graph_cache(graphs_folder='data/graphs', use_cached=False):
    for f in glob('{}/*'.format(graphs_folder)):
        if not os.path.isdir(f): continue
        is_graph_folder = os.path.isdir('{}/all'.format(f)) or len(glob('{}/*0.gml'.format(f))) != 0
        if not is_graph_folder: continue
        print('Creating: {}'.format(f))
        dataset_helper.get_gml_graph_dataset(f, use_cached=use_cached)


def get_graphs_only(X) -> list:
    assert len(X)
    if isinstance(X[0], nx.Graph) or ((isinstance(X[0], tuple) and not isinstance(X[0][1], str))):
        return X
    assert isinstance(X[0], tuple)
    X_ = [x for x, _ in X]
    assert isinstance(X_[0], nx.Graph)
    return X_


def get_mutag_enzyme_graphs(dataset='MUTAG', as_adj=True):
    A, gr_id, graph_label, node_label = get_graph_benchmark_dataset(dataset)
    graph_2_idx = collections.defaultdict(list)
    for idx, x in enumerate(gr_id):
        graph_2_idx[x].append(idx)

    for graph, idxs in graph_2_idx.items():
        assert list(sorted(idxs)) == idxs

    graph_2_idx = {graph: (np.min(idxs), np.max(idxs)) for graph, idxs in graph_2_idx.items()}

    X, Y = [], []
    for idx, (graph, (min_, max_)) in enumerate(sorted(graph_2_idx.items(), key=lambda x: x[0])):
        assert idx == graph
        adj = A[min_:max_ + 1, min_:max_ + 1]
        labels = node_label[min_:max_ + 1]

        x = (adj, labels)
        X.append(x)

        y = graph_label[idx]
        Y.append(y)

    if not as_adj:
        X = convert_adjs_tuples_to_graphs(X)

    return X, Y


def get_graph_benchmark_dataset(dataset, folder='data/graph_only'):
    import scipy
    import scipy.sparse

    folder = '{}/{}'.format(folder, dataset)
    assert os.path.exists(folder)

    def get_file(x):
        return '{}/{}_{}'.format(folder, dataset, x)

    files = {}
    for name, file in [('A', 'A.txt'), ('graph_indicator', 'graph_indicator.txt'), ('graph_labels', 'graph_labels.txt'), ('node_attributes', 'node_attributes.txt'), ('node_labels', 'node_labels.txt')]:
        file = get_file(file)
        if not os.path.exists(file):
            continue
        with open(file) as f:
            files[name] = f.read().strip()

    # Convert A to sparse matrix
    rows, cols = [], []
    for line in files['A'].splitlines():
        row, col = [int(x.strip()) for x in line.split(',')]
        rows.append(row)
        cols.append(col)
    data = [1] * len(rows)
    A = scipy.sparse.csr_matrix((data, (rows, cols)), dtype=np.uint8)

    def get_as_array(x):
        return np.array([int(y.strip()) for y in files[x].splitlines()])

    # Graph indices
    graph_indices = get_as_array('graph_indicator') - 1

    # Graph labels
    graph_labels = get_as_array('graph_labels')
    assert np.max(graph_indices) == len(graph_labels) - 1

    # Node labels
    node_labels = get_as_array('node_labels')
    return A, graph_indices, graph_labels, node_labels

def get_all_graph_benchmark_dataset_names(has_node_label=True, folder='data/graph_only'):
    folders = [x for x in glob('{}/*'.format(folder)) if os.path.isdir(x) and len(glob('{}/*_A.txt'.format(x)))]

    if has_node_label:
        folders = [x for x in folders if len(glob('{}/*_node_labels.txt'.format(x)))]

    return list(sorted([x.rsplit('/', 1)[-1] for x in folders]))