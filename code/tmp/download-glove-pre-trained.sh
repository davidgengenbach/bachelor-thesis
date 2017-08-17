#!/usr/bin/env sh

set -x

CONVERT_TO_WORD2VEC=1
FOLDER="glove"

cd $FOLDER

for pre_trained in glove.6B.zip glove.42B.300d.zip glove.840B.300d.zip glove.twitter.27B.zip; do
    wget "http://nlp.stanford.edu/data/$pre_trained"
    unzip $pre_trained
    if [ "$CONVERT_TO_WORD2VEC" == 1 ]; then
        continue
        # TODO
        python -m gensim.scripts.glove2word2vec –input "$pre_trained" -output "$OUT"
    fi
done

