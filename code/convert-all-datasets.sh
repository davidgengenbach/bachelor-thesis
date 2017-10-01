#!/usr/bin/env bash

#set -x

for i in $(python -c "for i in [i for i in __import__('dataset_helper').get_all_available_dataset_names() if 'ana' not in i and i != 'small' and i != 'r8']: print(i)"); do
    echo "############### Preparing dataset: $i"
    ./convert-datasets.py --train_size 1 --dataset_name $i --one_document_per_folder --force --preprocess
done