#!/usr/bin/env python3
"""Converts a dataset for concept-graph extraction.

out format:
    creates folders for train/test and folder for each class with the documents as .txt:
    - {out_folder}
        |_ topics.tsv
        |_ train
            |_ class1
                |_ doc1.txt
                |_ doc2.txt
            |_ class2
                |_ doc3.txt
                |_ doc4.txt
"""
import numpy as np
import codecs
import os
import pickle
from time import time
import re

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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
    parser.add_argument('--dataset_name',
                        type=str,
                        default='ng20')
    parser.add_argument('--one_document_per_folder',
                        action='store_true')
    parser.add_argument('--rename',
                        action='store_true')
    parser.add_argument('--out_folder',
                        type=str,
                        default='../extract-concept-graphs/code/data')
    args = parser.parse_args()
    assert args.train_size > 0 and args.train_size <= 1
    return args

def main():
    args = vars(get_args())
    process(**args)

def preprocess_text(text):
    reg = r'\n\n+'
    text = '\n'.join(x.strip() for x in text.split('\n') if x.strip() != '')
    text = re.sub('\n', ' ', re.sub(reg, '//', text)).replace('//', '\n')
    return text

def process(dataset_name, out_folder, train_size, random_state_for_shuffle, one_document_per_folder, rename, max_elements):
    TOPIC_ID_OFFSET = 100
    data = get_topics_from_sklearn()
    for idx, a in enumerate(data):
        data[idx] = (a[0], preprocess_text(a[1]))
    Y = [x[0] for x in data]
    X = [x[1] for x in data]
    topics = get_by_topics(X, Y)

    topics_count = {topic: len(docs) for topic, docs in topics.items()}
    out_folder = os.path.join(out_folder, dataset_name)
    
    os.makedirs(out_folder, exist_ok = True)

    # Map from topics to their ids
    topics_2_id = {topic: TOPIC_ID_OFFSET + idx for idx, topic in enumerate(topics.keys())}

    # Save topics_2_ids tab-seperated
    with codecs.open(os.path.join(out_folder, 'topics.tsv'), 'w') as f:
        for topic, idx in sorted(topics_2_id.items(), key = lambda x: x[1]):
            f.write('{}\t{}\n'.format(idx, topic))

    for topic, docs in topics.items():
        topic_id = topics_2_id[topic] if rename else topic

        # Create train/test split
        if train_size == 1.0:
            docs_train, docs_test = shuffle(
                docs, random_state=random_state_for_shuffle
            ), []
        else:
            docs_train, docs_test = train_test_split(
                docs,
                train_size=train_size,
                random_state=random_state_for_shuffle
            )
        if max_elements != -1:
            max_elements_train = min(int(max_elements * train_size), len(docs_train))
            max_elements_test = min(int(max_elements * (1 - train_size)), len(docs_test))
            docs_train = docs_train[:max_elements_train]
            docs_test = docs_test[:max_elements_test]
            

        assert len(docs_train) > 0, "\t-> len(docs_train) == 0"
        assert train_size == 1.0 or len(docs_test) > 0, "\t-> len(docs_test) == 0"

        print('Category: {:<27} #docs: train {:>3}, test {:>3}'.format('"' + topic + '"', len(docs_train), len(docs_test)))
        
        # Save sets
        for doc_set_name, doc_set in [('train', docs_train), ('test', docs_test)]:
            # Create set folder if not one_document_per_folder
            if one_document_per_folder:
                dataset_folder = os.path.join(out_folder, '{}')
                folder = dataset_folder.format(doc_set_name)
            else:
                dataset_folder = os.path.join(out_folder, '{}', str(topic_id))
                folder = dataset_folder.format(doc_set_name)
            os.makedirs(folder, exist_ok=True)
            for idx, doc in enumerate(doc_set):
                doc_id = str(idx).zfill(4)
                if one_document_per_folder:
                    filename = '{}/{}_{}/{}.txt'.format(folder, topic_id, doc_id, '0')
                    os.makedirs(os.path.join(*filename.split('/')[:-1]), exist_ok=True)
                else:
                    filename = '{}/{}.txt'.format(folder, str(idx).zfill(4))
                with codecs.open(filename, 'w') as f:
                    f.write(doc)

def get_topics_from_sklearn(subset = 'all', remove = ('headers', 'footers', 'quotes'), categories = None):
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(subset = subset, remove = remove, categories = categories)
    target_names = data.target_names
    return [(target_names[target], doc) for target, doc in zip(data.target, data.data)]

def get_by_topics(X, Y):
    """Returns a dictionary where the keys are the topics, the values are the documents of that topic.

    Args:
        X (list of list of str): The documents
        Y (list of str): The topics for the topics. len(X) == len(Y)

    Returns:
        dict: Keys are topics, values are the corresponding docs
    """
    assert len(X) == len(Y)
    assert len(set(Y)) > 0

    topics = {x: [] for x in set(Y)}
    for clazz, words in zip(Y, X):
        if words.strip() != '':
            topics[clazz].append(words)
    return topics


def get_topic_file(file):
    """Reads in a file of documents. Each line corresponds to a document.
    On each line, the first part is the topic of that document. The doc follows after a "\t".

    Args:
        file (str): The file

    Returns:
        list: returns a list of lists. One row: list('topic', 'word1 word2')
    """
    with codecs.open(file, 'r') as f:
        data = f.read().split('\n')
        data = [x.strip().split('\t') for x in data if x.strip() != '' and len(x.strip().split('\t')) == 2]
        return data


if __name__ == '__main__':
    main()




# UNUSED
def get_topics(in_file, use_pickle=True):
    """Returns a dict with the docs of the topics. Retrieves either as a pickle of converts it.

    Args:
        in_file (str): The dataset file

    Returns:
        dict: Keys are topics, values are the corresponding values
    """
    pickled_topics = in_file + '.topics.npy'
    if use_pickle and os.path.exists(pickled_topics):
        with open(pickled_topics, 'rb') as f:
            data = pickle.load(f)
    else:
        data = get_topic_file(in_file)
        
        with open(pickled_topics, 'wb') as f:
            pickle.dump(data, f)
        return data
    return data