#!/usr/bin/env python

import os
import pickle
from joblib import delayed, Parallel
from time import time

from misc import w2v_d2v
from preprocessing import preprocessing
from utils import dataset_helper, helper, time_utils
from utils.logger import LOGGER


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Create word2vec embeddings for datasets')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--n_jobs_w2v', type=int, default=1)
    parser.add_argument('--n_jobs_spacy', type=int, default=1)
    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument('--embedding_iter', type=int, default=30)
    parser.add_argument('--embedding_min_count', type=int, default=0)
    parser.add_argument('--embedding_save_path', type=str, default="data/embeddings/trained")
    parser.add_argument('--limit_dataset', nargs='+', type=str, default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    helper.print_script_args_and_info(args)

    limited_datasets = args.limit_dataset
    os.makedirs(args.embedding_save_path, exist_ok=True)

    datasets = dataset_helper.get_dataset_names_with_concept_map(limit_datasets=limited_datasets)
    Parallel(n_jobs=args.n_jobs)(delayed(process_dataset)(dataset_name, args) for dataset_name in datasets)

    LOGGER.info('Finished')


def process_dataset(dataset_name, args):
    try:
        embeddings_file = '{}/{}.npy'.format(args.embedding_save_path, dataset_name)
        if not args.force and os.path.exists(embeddings_file): return

        start_time = time()
        LOGGER.info('{:30} Starting'.format(dataset_name))

        X, Y = dataset_helper.get_dataset(dataset_name=dataset_name)
        X = preprocessing.preprocess_text_spacy(X, n_jobs=args.n_jobs_spacy)
        X = [[word.text.lower().strip() for word in doc] for doc in X]

        model = w2v_d2v.train_w2v(
            X,
            min_count=args.embedding_min_count,
            size=args.embedding_size,
            iter=args.embedding_iter,
            workers=args.n_jobs_w2v
        )

        word_vectors = model.wv
        del model

        with open(embeddings_file, 'wb') as f:
            pickle.dump(word_vectors, f)

        duration_in_s = time_utils.seconds_to_human_readable(time() - start_time)
        LOGGER.info('{:30} Finished (time={})'.format(dataset_name, duration_in_s))
    except Exception as e:
        LOGGER.exception(e)


if __name__ == '__main__':
    main()
