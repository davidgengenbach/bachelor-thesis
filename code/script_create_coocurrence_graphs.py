#!/usr/bin/env python

import os
import pickle

import spacy
from joblib import delayed, Parallel

from preprocessing import preprocessing
from utils import dataset_helper, graph_helper, helper, time_utils
from time import time


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Calculates the cooccurrence matrices and saves them')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--n_jobs_coo', type=int, default=1)
    parser.add_argument('--min_length', type=int, default=2)
    parser.add_argument('--window_size_start', type=int, default=1)
    parser.add_argument('--window_size_end', type=int, default=4)
    parser.add_argument('--lemmatize', action = 'store_true')
    parser.add_argument('--force', action = 'store_true')
    parser.add_argument('--limit_dataset', nargs='+', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    start_time = time()
    helper.print_script_args_and_info(args)

    Parallel(n_jobs=args.n_jobs)(delayed(process_dataset)(dataset_name, args) for dataset_name in dataset_helper.get_all_available_dataset_names(limit_datasets=args.limit_dataset))

    print('Finished (time={})'.format(time_utils.seconds_to_human_readable(time() - start_time)))

def process_dataset(dataset, args):
    start_time = time()
    X, Y = dataset_helper.get_dataset(dataset)

    min_length = args.min_length
    lemmatize = args.lemmatize

    def doc_filter(doc):
        if not only_nouns and min_length == -1:
            return doc
        return [word for word in doc if (not only_nouns or word.pos == spacy.parts_of_speech.NOUN) and (word.text.strip() != '') and (min_length == -1 or len(word) >= args.min_length)]

    X_preprocessed = preprocessing.preprocess_text_spacy(X, n_jobs=args.n_jobs_coo)
    
    for window_size in range(args.window_size_start, args.window_size_end):
        for lemmatize in set([False, lemmatize]):
            for only_nouns in [False, True]:
                cache_file = '{}/dataset_graph_cooccurrence_{}_{}_{}_{}.npy'.format(
                        dataset_helper.CACHE_PATH,
                        window_size,
                        'only-nouns' if only_nouns else 'all',
                        'lemmatized' if lemmatize else 'un-lemmatized',
                        dataset
                )

                if not args.force and os.path.exists(cache_file):
                    continue

                X_filtered = [doc_filter(doc) for doc in X_preprocessed]

                if lemmatize:
                    X_filtered = [[word.lemma_ for word in doc] for doc in X_filtered]

                X_processed = graph_helper.convert_dataset_to_co_occurence_graph_dataset(
                    X_filtered,
                    window_size = window_size,
                    n_jobs = args.n_jobs_coo
                )

                with open(cache_file, 'wb') as f:
                    pickle.dump((X_processed, Y), f)

    print('{:30} Finished (time={})'.format(dataset, time_utils.seconds_to_human_readable(start_time - time())))


if __name__ == '__main__':
    main()