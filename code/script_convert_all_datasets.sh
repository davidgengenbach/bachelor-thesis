#!/usr/bin/env bash


for i in ling-spam ng20 reuters-21578 review_polarity rotten_imdb tagmynews webkb; do
    echo "##### Preparing dataset: '$i'"
    ./script_convert_datasets.py --train_size 1 --dataset_name $i --one_document_per_folder --preprocess #--force
done