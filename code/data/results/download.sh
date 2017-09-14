#!/usr/bin/env bash

folder_name="$(date "+%Y-%m-%d_%H:%M")"

find . -name '*.npy' -depth 1 -exec rm {} \;
find predictions -name '*.npy' -depth 1 -exec rm {} \;

rm -rf $folder_name

mkdir -p "$folder_name"
cd "$folder_name"

echo "Results"
ssh ba sh get_results.sh > /dev/null 2>&1
scp ba:results.zip .
echo "- Starting to unzip results"
unzip -q results.zip
rm -f ../*.npy

echo "Predictions"
scp ba:predictions.zip .
echo "- Starting to unzip predictions"
unzip -q predictions.zip

./copy "$folder_name"
