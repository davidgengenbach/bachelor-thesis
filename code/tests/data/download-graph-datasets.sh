#!/usr/bin/env bash

# MUTAG and ENZYMES

wget -O tmp.zip 'https://ndownloader.figshare.com/articles/899875/versions/1'
unzip tmp.zip
rm tmp.zip

# ...
for i in *.zip; do
	unzip $i
done

rm *.zip
