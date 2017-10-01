#!/usr/bin/env python3
"""Converts a dataset for concept-graph extraction."""
import codecs
import importlib
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from time import time
import numpy as np
from preprocessing import preprocessing
from utils import git_utils, time_utils, dataset_helper


def get_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='convert dataset for concept-graph extraction. See this file for more info')
    parser.add_argument('--train_size',
                        type=float,
                        help="The percentage of data that is used for train, rest is used for test",
                        default=0.8)
    parser.add_argument('--random_state_for_shuffle',
                        type=int,
                        default=42)
    parser.add_argument('--max_elements',
                        type=int,
                        default=-1)
    parser.add_argument('--concat_train_instances',
                        action='store_true')
    parser.add_argument('--dataset_name',
                        type=str,
                        default='ng20')
    parser.add_argument('--one_document_per_folder',
                        action='store_true')
    parser.add_argument('--rename',
                        action='store_true')
    parser.add_argument('--out_folder',
                        type=str,
                        default='data/datasets-prepared')
    parser.add_argument('--dataset_folder',
                        type=str,
                        default='data/datasets')
    parser.add_argument('--shuffle',
                        action='store_true')
    parser.add_argument('--preprocess',
                        action='store_true')
    parser.add_argument('--force',
                        action='store_true')
    args = parser.parse_args()
    assert args.train_size > 0 and args.train_size <= 1
    return args


def main():
    args = vars(get_args())
    process(**args, args=args)


def process(dataset_name, out_folder, train_size, random_state_for_shuffle, one_document_per_folder, rename, max_elements, dataset_folder, force, args, concat_train_instances, shuffle, preprocess):

    out_folder = os.path.join(out_folder, dataset_name)

    if not force and os.path.isdir(out_folder):
        print('Outfolder existing! Aborting ({})'.format(out_folder))
        sys.exit(1)

    X, Y = dataset_helper.get_dataset(dataset_name, dataset_folder)
    
    print('#Docs: {}'.format(len(X)))
    
    if preprocess:
        X = preprocessing.concept_map_preprocessing(X)
    
    if shuffle:
        data_train_X, data_test_X, data_train_Y, data_test_Y = dataset_helper.split_dataset(X, Y, random_state_for_shuffle = random_state_for_shuffle, train_size = train_size)
    else:
        data_train_X, data_test_X, data_train_Y, data_test_Y = X, [], Y, []

    if train_size == 1.0:
        sets = [
            ('all', data_train_X, data_train_Y)
        ]
    else:
        sets = [
            ('train', data_train_X, data_train_Y),
            ('test', data_test_X, data_test_Y)
        ]

    # Create folder
    os.makedirs(out_folder, exist_ok=True)
    all_topic_counts = defaultdict(int)
    for set_name, X, Y in sets:
        topic_id_counters = defaultdict(int)
        set_folder = os.path.join(out_folder, set_name)
        assert len(X) == len(Y)

        for x, y in zip(X, Y):
            # Create set folder if not one_document_per_folder
            if one_document_per_folder:
                folder = set_folder
            else:
                folder = os.path.join(set_folder, str(y))
            os.makedirs(folder, exist_ok=True)

            doc_id = str(topic_id_counters[y]).zfill(4)

            if concat_train_instances and set_name == 'train':
                filename = '{}/{}/{}.txt'.format(folder, y, doc_id)
            elif one_document_per_folder:
                filename = '{}/{}_{}/{}.txt'.format(folder, y, doc_id, '0')
            os.makedirs(os.path.join(*filename.split('/')[:-1]), exist_ok=True)

            with codecs.open(filename, 'w') as f:
                f.write(x)
            all_topic_counts[y] += 1
            topic_id_counters[y] += 1

    with open(os.path.join(out_folder, 'stats.json'), 'w') as f:
        json.dump({
            'total_docs': sum(all_topic_counts.values()),
            'categories': list(set(Y)),
            'topic_counts': all_topic_counts,
            'set_counts': {name: len(X) for name, X, Y in sets},
            'params': args,
            'timestamp': time_utils.get_time_formatted(),
            'unix_timestamp': time_utils.get_timestamp(),
            'git_commit': str(git_utils.get_current_commit())
        }, f, indent=4, sort_keys = True)


if __name__ == '__main__':
    main()
