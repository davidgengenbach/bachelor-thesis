import networkx as nx
from transformers.nx_graph_to_tuple_transformer import NxGraphToTupleTransformer


tuple_trans = NxGraphToTupleTransformer()

def get_random_test_graph(num_nodes=5, seed=42):
    return nx.erdos_renyi_graph(num_nodes, p=0.5, seed=seed)


def get_test_graph(num_nodes=5):
    return nx.complete_graph(num_nodes)


def get_complete_graphs(num_graphs, as_tuples = False, num_nodes=5):
    graphs = [get_test_graph(num_nodes) for i in range(num_graphs)]

    if as_tuples:
        tuple_trans.transform(graphs)

    return graphs
