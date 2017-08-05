#!/usr/bin/env bash

set -x

for i in $(python -c "for i in __import__('dataset_helper').get_all_available_dataset_names(): print(i)"); do
    echo "############### Preparing dataset: $i"
    ./convert-datasets.py --dataset_name $i --force --concat_train_instances --one_document_per_folder
    exit
done