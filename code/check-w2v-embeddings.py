#!/usr/bin/env python

from glob import glob
import json
import dataset_helper
import graph_helper
from joblib import delayed, Parallel

def process_dataset(dataset_name, args, embedding_models):
    print('Processing: {}'.format(dataset_name))
    results = {}
    used_models = embedding_models + [('trained', dataset_helper.get_w2v_embedding_for_dataset(dataset_name))] if args.check_own_embeddings else embedding_models
    all_graph_cache_files = [x for x in dataset_helper.get_all_cached_graph_datasets() if dataset_name in x]
    graph_cache_files = []
    found_all_cache = False
    found_gml_cache = False
    for cache_file in all_graph_cache_files:
        # ...
        if len(graph_cache_files) == 2: break
        if (not found_gml_cache and 'gml' in cache_file) or (not found_all_cache and 'all' in cache_file):
            found_all_cache = found_all_cache or 'all' in cache_file
            found_gml_cache = found_gml_cache or 'gml' in cache_file
            graph_cache_files.append(cache_file)

    if len(graph_cache_files) != 2:
            print('\tFound: gml: {}, all: {}'.format(found_gml_cache, found_all_cache))
    for model_name, model in used_models:
        results[model_name] = {}
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
            results[model_name][graph_cache_file] = {
                'num_labels': len(labels),
                'counts': counter
            }
    return results

def main():
    args = get_args()

    print('Args: {}'.format(args))

    embedding_models = []
    # GloVe embeddings
    if args.check_glove:
        embedding_models += [(x, get_embedding_model(x, binary = False)) for x in glob(args.glove_files)]

    if args.check_google_news:
        embedding_models.append((args.google_news_file, get_embedding_model(args.google_news_file, binary = True)))

    all_datasets = dataset_helper.get_all_available_dataset_names()

    # ...
    embedding_results = {dataset_name: result for dataset_name, result in zip(all_datasets, Parallel(n_jobs=args.n_jobs)(delayed(process_dataset)(dataset_name, args, embedding_models) for dataset_name in all_datasets))}
        
    with open(args.results_file, 'w') as f:
        json.dump(embedding_results, f)
    print('Saved results', json.dumps(embedding_results))

def get_embedding_model(w2v_file, binary = False, first_line_header = True):
    import gensim
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

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Checks whether the graph labels are in different embeddings models (like GoogleNews or GloVe)')
    parser.add_argument('--n_jobs', type=int, default = 1)
    parser.add_argument('--check_glove', action = 'store_true')
    parser.add_argument('--check_google_news', action = 'store_true')
    parser.add_argument('--check_own_embeddings', action = 'store_true')
    parser.add_argument('--glove_files', type = str, default = 'data/embeddings/glove/*.w2v.txt')
    parser.add_argument('--google_news_file', type = str, default = 'data/embeddings/GoogleNews-vectors-negative300.bin')
    parser.add_argument('--results_file', type = str, default = 'data/embeddings/results.json')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()