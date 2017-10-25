#!/usr/bin/env python

import os
import pickle

import spacy
from joblib import delayed, Parallel

from preprocessing import preprocessing
from utils import dataset_helper, graph_helper


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Calculates the cooccurrence matrices and saves them')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--n_jobs_coo', type=int, default=1)
    parser.add_argument('--window_size_start', type=int, default=1)
    parser.add_argument('--window_size_end', type=int, default=4)
    parser.add_argument('--min_length', type=int, default=-1)
    parser.add_argument('--lemmatize', action = 'store_true')
    parser.add_argument('--force', action = 'store_true')
    parser.add_argument('--limit_dataset', type=str, default = None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print('args=', args)
    Parallel(n_jobs=args.n_jobs)(delayed(process_dataset)(dataset_name, args) for dataset_name in dataset_helper.get_all_available_dataset_names())

def process_dataset(dataset, args):
    if args.limit_dataset and dataset != args.limit_dataset: return
    X, Y = dataset_helper.get_dataset(dataset)
    #X, Y = X[:1], Y[:1]

    print('dataset: {:15} - preprocessing'.format(dataset))
    X_preprocessed = preprocessing.preprocess_text_spacy(X, min_length=args.min_length, concat = False)
    
    for window_size in range(args.window_size_start, args.window_size_end):
        for only_nouns in [False, True]:
            cache_file = dataset_helper.CACHE_PATH + '/dataset_graph_cooccurrence_{}_{}_{}_{}.npy'.format(window_size, 'only-nouns' if only_nouns else 'all', 'lemmatized' if args.lemmatize else 'un-lemmatized', dataset)
            
            if not args.force and os.path.exists(cache_file):
                print('dataset: {:15} - cache file exists'.format(dataset, cache_file))
                continue

            print('dataset: {:15} - window_size={}, only_nouns={:<6} ({})'.format(dataset, window_size,  only_nouns, cache_file))

            if only_nouns:
                X_filtered = [[word for word in doc if (word.pos == spacy.parts_of_speech.NOUN) and (word.text.strip() != '')] for doc in X_preprocessed]
            else:
                X_filtered = X_preprocessed

            X_processed, Y_processed = graph_helper.convert_dataset_to_co_occurence_graph_dataset(X_filtered, Y, only_nouns = only_nouns, min_length = args.min_length, window_size = window_size, n_jobs = args.n_jobs_coo, preprocess = False, lemma_ = args.lemmatize)

            with open(cache_file, 'wb') as f:
                pickle.dump((X_processed, Y_processed), f)


if __name__ == '__main__':
    main()