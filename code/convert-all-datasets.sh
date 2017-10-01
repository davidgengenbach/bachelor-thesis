#!/usr/bin/env bash


for i in ling-spam ng20 reuters-21578 review_polarity rotten_imdb tagmynews webkb; do
    echo "############### Preparing dataset: $i"
    ./convert-datasets.py --train_size 1 --dataset_name $i --one_document_per_folder --force --preprocess
done

#set -x
#WHITELIST="['ling-spam', 'ng20', 'reuters-21578', 'review_polarity', 'rotten_imdb', 'tagmynews', 'webkb']"
#for i in $(python -c "for i in [i for i in __import__('dataset_helper').get_all_available_dataset_names() if i in $WHITELIST]: print(i)"); do