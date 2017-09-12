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
#cp *.npy ..
echo "- Copying results"
find . -name '*.npy' -depth 1 -type f | cut -c 3- | xargs -I "{}" cp "{}" "../"

echo "Predictions"
scp ba:predictions.zip .
echo "- Starting to unzip predictions"
unzip -q predictions.zip
cd predictions

echo "- Copying predictions"
# "Argument list too long" error on "mv *.npy"
cd predictions
find . -type f | cut -c 3- | xargs -I "{}" cp "{}" "../../predictions/{}"
cd ..
#mv predictions/* ../predictions
#rm -rf predictions