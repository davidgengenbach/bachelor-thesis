#!/usr/bin/env python

import dataset_helper
import preprocessing
import w2v_d2v
import pickle
import os
from joblib import delayed, Parallel

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Create w2v for datasets')
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument('--embedding_size', type=int, default=50)
    parser.add_argument('--embedding_iter', type=int, default=10)
    parser.add_argument('--embedding_min_count', type=int, default=0)
    parser.add_argument('--embedding_save_path', type=str, default="data/embeddings/trained")
    parser.add_argument('--force', action = 'store_true')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    Parallel(n_jobs=args.n_jobs)(delayed(process_dataset)(dataset_name, args) for dataset_name in dataset_helper.get_all_available_dataset_names())

    print('Finished')


def process_dataset(dataset_name, args):
    embeddings_file = '{}/{}.npy'.format(args.embedding_save_path, dataset_name)
    if not args.force and os.path.exists(embeddings_file):
        print('{:20} - Embedding file already exists. Skipping dataset'.format(dataset_name))
        return

    with open(embeddings_file, 'w') as f:
        f.write('NOT_DONE')

    print('{:20} - Retrieving dataset'.format(dataset_name))
    X, Y = dataset_helper.get_dataset(dataset_name=dataset_name)

    print('{:20} - Preprocessing'.format(dataset_name))
    X = preprocessing.preprocess_text_spacy(X, min_length=0, concat=False, only_nouns=False)
    X = [[word.text for word in doc] for doc in X]

    print('{:20} - Training'.format(dataset_name))
    model = w2v_d2v.train_w2v(
        X,
        min_count=args.embedding_min_count,
        size=args.embedding_size,
        iter=args.embedding_iter
    )
    word_vectors = model.wv
    del model
    print('{:20} - Saving'.format(dataset_name))
    with open(embeddings_file, 'wb') as f:
        pickle.dump(word_vectors, f)


if __name__ == '__main__':
    main()
