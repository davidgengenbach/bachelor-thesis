#!/usr/bin/env python3

from joblib import delayed, Parallel
import dataset_helper
import graph_helper
import embeddings

def get_embeddings_for_labels(labels, embedding, check_most_similar = False, restrict_vocab = None, lookup_embedding = None, topn = 20):
    assert not check_most_similar or lookup_embedding is not None
    not_found, embeddings = [], {}

    for label in labels:
        label = label.lower()
        if label in embedding:
            embeddings[label] = embedding[label]
        elif check_most_similar and label in lookup_embedding:
            most_similar = lookup_embedding.similar_by_word(label, topn = topn)
            most_similar_labels = [label for label, similarity in most_similar]
            match = set(most_similar_labels) & set(restrict_vocab)

            if len(match):
                embeddings[label] = embedding[list(match)[0]]
            else:
                not_found.append(label)
        else:
            not_found.append(label)
    return embeddings, not_found


def process_dataset(dataset_name, pre_trained_embedding, args):
    if dataset_name != 'ling-spam': return
    print('{:15} - Start'.format(dataset_name))

    all_words_graphs = [graph_cache_file for graph_cache_file in dataset_helper.get_all_cached_graph_datasets() if '_{}.'.format(dataset_name) in graph_cache_file and 'all' in graph_cache_file]
    if not len(all_words_graphs):
        print('{:15} - no all words graph! Aborting'.format(dataset_name))
        return

    all_words_graph = all_words_graphs[0]

    X, Y = dataset_helper.get_dataset_cached(all_words_graph)
    all_labels = graph_helper.get_all_node_labels(X)
    
    trained_embedding = dataset_helper.get_w2v_embedding_for_dataset(dataset_name)

    embeddings_trained_labels = set(trained_embedding.vocab.keys())
    not_found_trained = set(all_labels) - embeddings_trained_labels

    embeddings_pre_trained_labels = set(pre_trained_embedding.vocab.keys())
    not_found_pre_trained = set(all_labels) - embeddings_pre_trained_labels

    in_both = embeddings_trained_labels & embeddings_pre_trained_labels

    embeddings_pre_trained, not_found_pre_trained_coreferenced = get_embeddings_for_labels(all_labels, pre_trained_embedding, check_most_similar = True, restrict_vocab = in_both, lookup_embedding = trained_embedding)
    
    print('Not found:\n\ttrained: {}\n\tpre_trained: {}\n\tafter_coreference: {}'.format(len(not_found_trained), len(not_found_pre_trained), len(not_found_pre_trained_coreferenced)))

    print('{:15} - Finish'.format(dataset_name))


def main():
    args = get_args()

    print('Loading pre-trained embedding')
    pre_trained_embedding = embeddings.get_embedding_model(args.pre_trained_embedding, binary = False, first_line_header = True, with_gensim = True)

    print('Starting to process datasets')
    Parallel(n_jobs=args.n_jobs)(delayed(process_dataset)(dataset_name, pre_trained_embedding, args) for dataset_name in dataset_helper.get_all_available_dataset_names())
    print('Finished')

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='tbd')
    parser.add_argument('--n_jobs', type=int, default = 1)
    parser.add_argument('--pre_trained_embedding', type=str, default = 'data/embeddings/glove/glove.6B.50d.w2v.txt')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()