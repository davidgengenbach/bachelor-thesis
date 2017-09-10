#!/usr/bin/env bash

folder_name="$(date "+%Y-%m-%d_%H:%M")"

rm -f *.npy
rm -f predictions/*.npy

mkdir "$folder_name"
cd $folder_name

ssh ba sh get_results.sh
scp ba:results.zip .
unzip results.zip
rm -f ../*.npy
cp *.npy ..

scp ba:predictions.zip .
unzip predictions.zip
mv predictions/* ../predictions
rm -rf predictions