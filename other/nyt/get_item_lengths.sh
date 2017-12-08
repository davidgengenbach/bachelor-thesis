#!/usr/bin/env bash

echo > data/word_lengths.txt
FILES=$(find data/unpacked -type f -name '*.xml')
for i in $FILES; do
    out=$(echo $i | cut -d'/' -f3-)
    out="$out,$(grep -Eo 'item-length="(.+?)" ' $i)"
    echo "$out" >> data/word_lengths.txt
done