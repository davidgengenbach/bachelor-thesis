#!/usr/bin/env python

import gensim
import dataset_helper
import graph_helper
from glob import glob

CHECK_GLOVE = False
CHECK_GOOGLE_NEWS = False
CHECK_OWN_EMBEDDINGS = True
GLOVE_FILES = 'data/embeddings/glove/*.w2v.txt'
GOOGLE_NEWS_W2V_FILE = 'data/embeddings/GoogleNews-vectors-negative300.bin'

def main():
    embedding_models = []
    # GloVe embeddings
    if CHECK_GLOVE:
        embedding_models += [(x, get_embedding_model(x, binary = False)) for x in glob(GLOVE_FILES)]

    if CHECK_GOOGLE_NEWS:
        embedding_models.append((GOOGLE_NEWS_W2V_FILE, get_embedding_model(GOOGLE_NEWS_W2V_FILE, binary = True)))

    for dataset_name in dataset_helper.get_all_available_dataset_names():
        print('Processing: {}'.format(dataset_name))
        used_models = embedding_models + [('trained', dataset_helper.get_w2v_embedding_for_dataset(dataset_name))] if CHECK_OWN_EMBEDDINGS else embedding_models
        all_graph_cache_files = [x for x in dataset_helper.get_all_cached_graph_datasets() if dataset_name in x]
        graph_cache_files = []
        found_all_cache = False
        found_gml_cache = False
        for cache_file in all_graph_cache_files:
            if len(graph_cache_files) == 2: break
            if (not found_gml_cache and 'gml' in cache_file) or (not found_all_cache and 'all' in cache_file):
                found_all_cache = found_all_cache or 'all' in cache_file
                found_gml_cache = found_gml_cache or 'gml' in cache_file
                graph_cache_files.append(cache_file)

        if len(graph_cache_files) != 2:
                print('\tDid not find: gml: {}, all: {}'.format(found_gml_cache, found_all_cache))
        for model_name, model in used_models:
            print('\tModel: {}'.format(model_name))
            for graph_cache_file in graph_cache_files:
                print('\t\tGraph: {}'.format(graph_cache_file))
                X, Y = dataset_helper.get_dataset_cached(graph_cache_file)
                labels = graph_helper.get_all_node_labels(X)
                print('\t\t#unique labels: {}'.format(len(labels)))
                counter = {'found': 0, 'not_found': 0}
                for idx, label in enumerate(labels):
                    if label in model:
                        counter['found'] += 1
                    else:
                        counter['not_found'] += 1
                print('\t\t{}, Found: {}%'.format(counter, int(100 * counter['found'] / len(labels))))

def get_embedding_model(w2v_file, binary = False, first_line_header = True):
    if binary:
        embeddings = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=True)
    else:
        embeddings = {}
        with open(w2v_file) as f:
            if first_line_header:
                first_line = f.readline()
                num_labels, num_dim = [int(x) for x in first_line.split(' ')]
            embeddings = {x.split(' ', 1)[0].strip(): x.split(' ', 1)[1].strip() for x in f}
    return embeddings

if __name__ == '__main__':
    main()