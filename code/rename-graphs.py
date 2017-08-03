#!/usr/bin/env python

import glob
import os
import sys
import shutil
import dataset_helper
import preprocessing
from joblib import Parallel, delayed

def sort_key(x):
    return x[0] + x[1]

def get_docs(folder):
    docs = []
    for file in glob.glob(folder + '/*/*.txt'):
        topic, file_id = file.split('/')[-2].rsplit('_', 1)
        with open(file) as f:
            docs.append((topic, file_id, f.read()))
    return sorted(docs, key = sort_key)

def get_graph_filelist(folder):
    def process(file):
        return file.split('/')[-1].replace('.gml', '').rsplit('_', 1)
    return sorted([process(file) for file in glob.glob(folder + '/*.gml')], key = sort_key)

#for DATASET in ['webkb', 'reuters-21578', 'ng20', 'ling-spam']:
for DATASET in ['ling-spam']:
    print('Processing: {}'.format(DATASET))
    dataset_name = DATASET
    folder_right_docs = 'data/datasets-prepared.bck/{}-single/all'.format(DATASET)
    folder_right_graphs = 'data/graphs/{}-single'.format(DATASET)

    topics = dataset_helper.get_dataset_dict(*dataset_helper.get_dataset(dataset_name, use_cached = True))
    right_docs = get_docs(folder_right_docs)
    right_graphs = get_graph_filelist(folder_right_graphs)
    print('Staring pre-processing', topics.keys())
    if DATASET != 'webkb':
        for topic, docs in topics.items():
            topics[topic] = Parallel(n_jobs=4)(delayed(preprocessing.preprocess_text_old)(text) for text in docs)
    else:
        for topic, docs in topics.items():
            topics[topic] = [preprocessing.preprocess_text_old(text) for text in docs]
            
    def strip(text):
        import re, string;
        pattern = re.compile('[\W_]+')
        return pattern.sub('', string.printable)

    import codecs
    mapping = {}
    for idx, (topic, docs) in enumerate(topics.items()):
        found, not_found = [], []
        mapping[topic] = {}
        m = mapping[topic]
        print(topic)
        candidates = []
        for doc_idx, doc in enumerate(docs):
            lower_doc = codecs.utf_8_encode(doc.lower())
            found_ = False
            for other_idx, (other_topic, other_file_id, other_doc) in enumerate(right_docs):
                if topic != other_topic: continue
                if doc.__hash__() == other_doc.__hash__() and doc == other_doc and other_file_id not in m:
                    m[other_file_id] = doc_idx
                    found.append(1)
                    found_ = True
                    break
            if not found_:
                candidates.append(doc_idx)
                not_found.append(0)
        if len(candidates):
            print('\tCandidates: {}'.format(candidates))
        for idx in candidates:
            original_doc = docs[idx]
            original_doc = strip(original_doc)
            for other_idx, (other_topic, other_file_id, other_doc) in enumerate(right_docs):
                if topic != other_topic: continue
                if original_doc == strip(other_doc) and other_file_id not in m:#[:min(10, len(other_doc))]:
                    m[other_file_id] = idx
                    found.append(1)
                    not_found.pop()
                    break
        assert len(not_found) == 0, 'Not all docs found! #not_found: {}'.format(len(not_found))
        assert len(found) == len(docs)
        print('{:<20} Found: {}'.format(topic, len(found)))
        
    # Copy graph folder
    new_folder_right_graphs = folder_right_graphs + '-RENAMED'
    shutil.rmtree(new_folder_right_graphs, ignore_errors= True)
    os.mkdir(new_folder_right_graphs)

    for other_topic, other_file_id in right_graphs:
        topic_mapping = mapping[other_topic]
        assert other_file_id in topic_mapping
        to_file_id = str(topic_mapping[other_file_id]).zfill(4)
        old_filename = os.path.join(folder_right_graphs, '{}_{}.gml'.format(other_topic, other_file_id))
        new_filename = os.path.join(new_folder_right_graphs, '{}_{}.gml'.format(other_topic, to_file_id))
        shutil.copy(old_filename, new_filename)
