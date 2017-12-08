#!/usr/bin/env bash

cd data

mkdir -p unpacked

files=$(find . -type f -name '*.tgz')
for i in $files; do
    echo $i
    FILE=$(basename $i)
    FOLDER=$(dirname $i | cut -d'/' -f2-)
    NEW_FOLDER="unpacked/${FOLDER}"
    mkdir -p $NEW_FOLDER
    tar zxf $i -C $NEW_FOLDER
done