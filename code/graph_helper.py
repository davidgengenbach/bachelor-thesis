import networkx as nx
from glob import glob
import os
import preprocessing
import cooccurrence
from joblib import Parallel, delayed

def convert_dataset_to_co_occurence_graph_dataset(X, Y, n_jobs = 4):
    #X, Y = X[:10], Y[:10]
    print('Pre-processing')
    X = preprocessing.preprocess_text_spacy(X, min_length = 2)
    print('Creating adjadency mats')
    mats = Parallel(n_jobs=n_jobs)(delayed(cooccurrence.get_coocurrence_matrix)(text) for text in X)
    print('Converting to networkx graphs')
    graphs = Parallel(n_jobs=n_jobs)(delayed(convert_from_numpy_to_nx)(*mat) for mat in mats)
    return graphs, Y


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


def get_graph_topic_stats(graphs_per_topic):
    df_graphs_per_topic = pd.DataFrame([(topic, len(graphs), [len(x.nodes()) for x in graphs], [len(x.edges()) for x in graphs]) for topic, graphs in graphs_per_topic.items(
    )], columns=['topic', 'num_graphs', 'num_nodes', 'num_edges']).set_index(['topic']).sort_values(by='num_graphs')
    df_graphs_per_topic['avg_nodes'] = df_graphs_per_topic.num_nodes.apply(lambda x: np.mean(x))
    df_graphs_per_topic['avg_edges'] = df_graphs_per_topic.num_edges.apply(lambda x: np.mean(x))
    return df_graphs_per_topic


def get_graphs_from_folder(folder, ext='gml', undirected=True, verbose=False):
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
    files = sorted(glob(folder + '/*' + ext))
    file_list = list(enumerate(files))
    for idx, graph_file in sorted(file_list):
        topic_and_id = graph_file.split('/')[-1].replace('.gml', '')
        topic = '_'.split(topic_and_id.split('_')[:-1])
        with open(graph_file) as f:
            graph_str = f.read()
        graph = get_gml_graph(graph_str, undirected)
        # Ignore empty graphs and graphs that could not be parsed
        if graph and graph.number_of_nodes() > 0 and graph.number_of_edges() > 0:
            X.append(graph)
            Y.append(topic)
        else:
            if graph is None:
                break
            if verbose:
                print("Empty graph: {}".format(topic_and_id))
            empty_graphs.append(topic_and_id)

    assert len(X) and len(Y), 'X or Y empty'
    assert len(X) == len(Y), 'X has not the same dimensions as Y'
    return X, Y


def get_gml_graph(graph_str, undirected=False, num_tries=5, verbose=False):
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
        if line.startswith('label'):
            next_line = graph_str_lines[idx + 1]
            label = next_line.replace('name', 'label')
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
                print('\tFound duplicate node label: "{}", trying to replace with first occurence'.format(duplicate_label))
            # Try to recover
            occurences = []
            for idx, line in enumerate(graph_str_lines):
                if line.startswith('label'):
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