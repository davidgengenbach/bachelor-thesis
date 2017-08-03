import networkx as nx


def convert_from_numpy_to_nx(word2id, id2word, mat):
    graph = nx.from_numpy_matrix(mat.toarray())
    nx.relabel_nodes(graph, mapping=id2word, copy=False)
    return graph


def get_graph_topic_stats(graphs_per_topic):
    df_graphs_per_topic = pd.DataFrame([(topic, len(graphs), [len(x.nodes()) for x in graphs], [len(x.edges()) for x in graphs]) for topic, graphs in graphs_per_topic.items(
    )], columns=['topic', 'num_graphs', 'num_nodes', 'num_edges']).set_index(['topic']).sort_values(by='num_graphs')
    df_graphs_per_topic['avg_nodes'] = df_graphs_per_topic.num_nodes.apply(lambda x: np.mean(x))
    df_graphs_per_topic['avg_edges'] = df_graphs_per_topic.num_edges.apply(lambda x: np.mean(x))
    return df_graphs_per_topic
