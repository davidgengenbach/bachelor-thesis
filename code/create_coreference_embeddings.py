#!/usr/bin/env python3

from joblib import delayed, Parallel
import dataset_helper
import graph_helper
import embeddings
import pickle
import coreference
from logger import LOGGER


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Creates embeddings for labels - given a pre-trained embedding model. Resolves missing labels with another embedding by searching for similar words for the missing word and using that similar word\'s embedding for the missing label')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--pre_trained_embedding', type=str, default='data/embeddings/glove/glove.6B.50d.w2v.txt')
    parser.add_argument('--embeddings_result_folder', type=str, default='data/embeddings/graph-embeddings')
    parser.add_argument('--limit_dataset', nargs='+', type=str, default=['ng20', 'ling-spam', 'reuters-21578', 'webkb'])
    parser.add_argument('--merge_threshold', nargs='+', type=float, default=[0.5, 0.7, 0.9])
    parser.add_argument('--topn', nargs='+', type=int, default=[1])
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()
    return args


def process_dataset(dataset_name, pre_trained_embedding, args):
    if args.limit_dataset and dataset_name not in args.limit_dataset:
        return

    LOGGER.info('{:15} - Start'.format(dataset_name))

    LOGGER.info('{:15} - Retrieving trained embedding'.format(dataset_name))

    trained_embedding = dataset_helper.get_w2v_embedding_for_dataset(dataset_name)

    graphs = dataset_helper.get_all_cached_graph_datasets(dataset_name)
    all_words_graphs = [g for g in graphs if 'all' in g]
    gml_graphs = [g for g in graphs if 'gml' in g]
    used_graphs  = []
    if len(all_words_graphs):
        used_graphs.append(all_words_graphs[0])
    if len(gml_graphs):
        used_graphs.append(gml_graphs[0])

    if not len(used_graphs):
        LOGGER.info('{:15} - no graphs found. Aborting'.format(dataset_name))
        return

    LOGGER.info('{:15} - Retrieving dataset'.format(dataset_name))
    all_labels = set()
    for graph_cache_file in used_graphs:
        X, _ = dataset_helper.get_dataset_cached(graph_cache_file)
        all_labels |= graph_helper.get_all_node_labels(X, as_sorted_list = False)

    LOGGER.info('{:15} - Resolving embeddings'.format(dataset_name))
    embeddings_pre_trained, not_found_pre_trained_coreferenced, not_found_trained, not_found_pre_trained, lookup, similar_els = embeddings.get_embeddings_for_labels_with_lookup(
        all_labels, trained_embedding, pre_trained_embedding)

    LOGGER.info('{:15} - Missing'.format(dataset_name))

    for label, s in [('trained', not_found_trained), ('pre_trained', not_found_pre_trained), ('after_coreference', not_found_pre_trained_coreferenced)]:
        LOGGER.info('\t{:20} {:>6}'.format(label, len(s)))

    embedding_file = '{}/{}.w2v.txt'.format(args.embeddings_result_folder, dataset_name)
    embeddings.save_embedding_dict(embeddings_pre_trained, embedding_file)
    embeddings_pre_trained = embeddings.load_word2vec_format(fname = embedding_file, binary = False)

    LOGGER.info('{:15} - Co-reference resolution'.format(dataset_name))
    max_topn = max(args.topn)
    
    similar_labels = coreference.get_most_similar_labels(all_labels, embeddings_pre_trained, max_topn)

    for topn in args.topn:
        for threshold in args.merge_threshold:
            LOGGER.info('{:15} - Co-reference resolution: topn: {}, threshold: {}'.format(dataset_name, topn, threshold))
            clique_lookup = coreference.create_label_cliques_by_similarity(similar_labels, threshold=threshold, topn=topn)

            new_lookup = embeddings.merge_lookups(clique_lookup, lookup)

            with open('{}/{}.threshold-{}.topn-{}.label-lookup.npy'.format(args.embeddings_result_folder, dataset_name, threshold, topn), 'wb') as f:
                pickle.dump(new_lookup, f)
    LOGGER.info('{:15} - Finished'.format(dataset_name))

def main():
    args = get_args()

    LOGGER.info('Loading pre-trained embedding')
    pre_trained_embedding = embeddings.get_embedding_model(
        args.pre_trained_embedding, binary=False, first_line_header=True, with_gensim=True)

    LOGGER.info('Starting to process datasets')
    Parallel(n_jobs=args.n_jobs)(delayed(process_dataset)(dataset_name, pre_trained_embedding, args) for dataset_name in dataset_helper.get_all_available_dataset_names())
    LOGGER.info('Finished')

if __name__ == '__main__':
    main()
