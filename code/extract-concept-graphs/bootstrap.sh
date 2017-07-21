#!/usr/bin/env sh

LIBRARY_PATH="libraries"
SEMILAR_FILE="SEMILAR-API-1.0.zip"
SEMILAR_FILE_MODELS="LSA-MODELS.zip"
GOOGLE_W2V_FILE="GoogleNews-vectors-negative300.bin.gz"

cd $LIBRARY_PATH

wget --continue -O "$GOOGLE_W2V_FILE" "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
# Extract

wget --continue -O "$SEMILAR_FILE" "http://deeptutor2.memphis.edu/Semilar-Web/public/downloads/SEMILAR-API-1.0.zip"
unzip $SEMILAR_FILE

wget --continue -O "$SEMILAR_FILE_MODELS" "http://deeptutor2.memphis.edu/Semilar-Web/public/downloads/LSA-MODELS.zip"
unzip $SEMILAR_FILE_MODELS
