#!/usr/bin/env bash

set -x

cd glove

for pre_trained in \
    glove.6B.zip \
    glove.42B.300d.zip \
    glove.840B.300d.zip \
    glove.twitter.27B.zip; do
    wget "http://nlp.stanford.edu/data/$pre_trained"
    unzip $pre_trained
done

