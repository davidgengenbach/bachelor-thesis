#!/usr/bin/env bash

SERVER='ba'

folder_name="$(date "+%Y-%m-%d_%H:%M")"

find . -name '*.npy' -depth 1 -exec rm {} \;
find predictions -name '*.npy' -depth 1 -exec rm {} \;

rm -rf $folder_name

mkdir -p "$folder_name"
cd "$folder_name"

echo "Results"
ssh $SERVER sh get_results.sh > /dev/null 2>&1
scp $SERVER:results.zip .
echo "- Starting to unzip results"
unzip -q results.zip
rm -f ../*.npy

echo "Predictions"
scp $SERVER:predictions.zip . 2> /dev/null
if [ -f 'predictions.zip' ]; then
    echo "- Starting to unzip predictions"
    unzip -q predictions.zip
else
    echo "- No predictions downloaded"
fi
#./copy "$folder_name"
