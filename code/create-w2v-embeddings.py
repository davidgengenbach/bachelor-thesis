#!/usr/bin/env python

import dataset_helper
import preprocessing
import w2v_d2v
import pickle
import os

for dataset_name in dataset_helper.get_all_available_dataset_names():
    print('Processing: {}'.format(dataset_name))
    embeddings_file = 'data/embeddings/{}.npy'.format(dataset_name)
    if os.path.exists(embeddings_file):
        print('\tEmbedding file already exists. Skipping dataset')
        continue
    
    with open(embeddings_file, 'w') as f:
        f.write(':)')

    X, Y = dataset_helper.get_dataset(dataset_name=dataset_name)
    X = [[word.lower() for word in w2v_d2v.tokenizer(doc) ] for doc in X]
    model = w2v_d2v.train_w2v(X)
    word_vectors = model.wv
    del model
    with open(embeddings_file, 'wb') as f:
        pickle.dump(word_vectors, f)
