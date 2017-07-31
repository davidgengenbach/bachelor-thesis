#!/usr/bin/env bash

for i in "ling-spam" "ng20" "reuters-21578" "webkb"; do
    echo "############### Preparing dataset: $i"
    ./convert-datasets.py --dataset_name $i --force --concat_train_instances --one_document_per_folder
done