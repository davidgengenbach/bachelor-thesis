#!/usr/bin/env bash

for file in *.dot; do
    filename=$(basename "$file")
    extension="${filename##*.}"
    filename="${filename%.*}"
    dot -Tpdf -o$filename.pdf $filename.dot
done