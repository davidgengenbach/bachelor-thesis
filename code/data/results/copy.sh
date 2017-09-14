#!/usr/bin/env bash

folder_name="$1"

if [ -z "$folder_name" ]; then
    folder_name=$(find . -type d -depth 1 -name '2017*' | sort -r | head -n1)
fi

echo "Copying from: $folder_name"

echo '- Removing old results and predictions'
find . -name '*.npy' -depth 1 -exec rm {} \;
find predictions -name '*.npy' -depth 1 -exec rm {} \;

cd $folder_name

echo "- Copying results"
find . -name '*.npy' -depth 1 -type f | cut -c 3- | xargs -I "{}" cp "{}" "../"

echo "- Copying predictions"
cd predictions
find . -type f | cut -c 3- | xargs -I "{}" cp "{}" "../../predictions/{}"
