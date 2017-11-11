from utils import dataset_helper, graph_helper
import deepwalk
from deepwalk import graph
from deepwalk import walks as serialized_walks
from gensim.models import Word2Vec
from deepwalk.skipgram import Skipgram
from gensim.models import Word2Vec
from misc import tsne
import random

max_memory_data_size = 1000000000
number_walks = 1000
representation_size = 64
seed = 0
undirected = True
vertex_freq_degree = False
walk_length = 60
window_size = 10
workers = 1
output = 'data/DUMP'

for dataset in dataset_helper.get_all_available_dataset_names():
    cache_file = dataset_helper.CACHE_PATH + '/dataset_graph_cooccurrence_{}.npy'.format(dataset)
    X, Y = dataset_helper.get_dataset(dataset, preprocessed=False, use_cached=True, transform_fn=graph_helper.convert_dataset_to_co_occurence_graph_dataset, cache_file=cache_file)
    break

models = []
for idx, g in enumerate(X):
    if idx == 3: break
    print('Graph: {:>4}'.format(idx))
    G = graph.from_networkx(g)

    print("Number of nodes: {}".format(len(G.nodes())))
    if len(G.nodes()) == 0:
        continue

    num_walks = len(G.nodes()) * number_walks

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * walk_length

    print("Data size (walks*length): {}".format(data_size))

    print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=number_walks, path_length=walk_length, alpha=0, rand=random.Random(seed))
    print("Training...")
    model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, workers=workers)

    # model.wv.save_word2vec_format(output)
    models.append(model)
print('Finished')