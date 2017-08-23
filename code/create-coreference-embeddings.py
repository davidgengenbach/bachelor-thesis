#!/usr/bin/env python3

from joblib import delayed, Parallel
import dataset_helper
import graph_helper
import embeddings
import pickle
import coreference


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Creates embeddings for labels - given a pre-trained embedding model. Resolves missing labels with another embedding by searching for similar words for the missing word and using that similar word\'s embedding for the missing label')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--pre_trained_embedding', type=str, default='data/embeddings/glove/glove.6B.50d.w2v.txt')
    parser.add_argument('--embeddings_result_folder', type=str, default='data/embeddings/graph-embeddings')
    parser.add_argument('--limit_dataset', type=str, default=None)
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()
    return args


def process_dataset(dataset_name, pre_trained_embedding, args):
    if args.limit_dataset and dataset_name != args.limit_dataset:
        return

    print('{:15} - Start'.format(dataset_name))

    print('{:15} - Retrieving trained embedding'.format(dataset_name))

    trained_embedding = dataset_helper.get_w2v_embedding_for_dataset(dataset_name)

    all_words_graphs = [graph_cache_file for graph_cache_file in dataset_helper.get_all_cached_graph_datasets(
    ) if '_{}.'.format(dataset_name) in graph_cache_file and 'all' in graph_cache_file]

    if not len(all_words_graphs):
        print('{:15} - no all-words graph found. Aborting'.format(dataset_name))
        return

    all_words_graph = all_words_graphs[0]

    print('{:15} - Retrieving dataset'.format(dataset_name))
    X, Y = dataset_helper.get_dataset_cached(all_words_graph)
    all_labels = graph_helper.get_all_node_labels(X)

    print('{:15} - Resolving embeddings'.format(dataset_name))
    embeddings_pre_trained, not_found_pre_trained_coreferenced, not_found_trained, not_found_pre_trained, lookup = embeddings.get_embeddings_for_labels_with_lookup(
        all_labels, trained_embedding, pre_trained_embedding)

    print('{:15} - Missing'.format(dataset_name))

    for label, s in [('trained', not_found_trained), ('pre_trained', not_found_pre_trained), ('after_coreference', not_found_pre_trained_coreferenced)]:
        print('\t{:20} {:>6}'.format(label, len(s)))

    with open('{}/{}.label-lookup.npy'.format(args.embeddings_result_folder, dataset_name), 'wb') as f:
        pickle.dump(lookup, f)

    embeddings.save_embedding_dict(
        embeddings_pre_trained, '{}/{}.w2v.txt'.format(args.embeddings_result_folder, dataset_name))

    print('{:15} - Finish'.format(dataset_name))


def main():
    args = get_args()

    print('Loading pre-trained embedding')
    pre_trained_embedding = embeddings.get_embedding_model(
        args.pre_trained_embedding, binary=False, first_line_header=True, with_gensim=True)

    print('Starting to process datasets')
    Parallel(n_jobs=args.n_jobs)(delayed(process_dataset)(dataset_name, pre_trained_embedding, args)
                                 for dataset_name in dataset_helper.get_all_available_dataset_names())
    print('Finished')

if __name__ == '__main__':
    main()
