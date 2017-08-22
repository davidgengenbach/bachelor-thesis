#!/usr/bin/env bash

LOG_PATH=~/logs

# Co-occurrence
NAME=create-coocurrence-graphs
python -u $NAME.py --n_jobs=8 --force > $LOG_PATH/$NAME.log 2>&1

# Phi cache
NAME=create-phi-graph-cache
python -u $NAME.py --force > $LOG_PATH/$NAME.log 2>&1

# Co-reference
NAME=create-coreference-embeddings
python -u $NAME.py --force > $LOG_PATH/$NAME.log 2>&1

# w2v
NAME=create-w2v-embeddings
python -u $NAME.py --force > $LOG_PATH/$NAME.log 2>&1

# check w2v
NAME=check-w2v-embeddings
python -u $NAME.py --check_glove --check_google_news --check_own_embeddings  > $LOG_PATH/$NAME.log 2>&1

# phi cache
NAME=create-phi-graph-cache
python -u  $NAME.py --n_jobs=6 --force > ~/logs/$NAME.log 2>&1