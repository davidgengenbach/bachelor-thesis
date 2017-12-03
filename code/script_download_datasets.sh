#!/usr/bin/env bash

#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


cd data/datasets
DIR="$(pwd -P)"

download_and_unzip() {
    NAME="$1"
    LINK="$2"
    FILE="$3"
    BACK_DIR="$4"

    folder="$NAME/src"
    mkdir -p $folder
    cd $folder

    wget --continue -O "$FILE" "$LINK" || true
    tar -zxvf "$FILE" || gunzip -d "$FILE" || unzip "$FILE" || echo "Warning: could not unzip/gunzip or un-tar $FILE"
    cd "$BACK_DIR"
}

download_and_unzip 'webkb' 'http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/webkb-data.gtar.gz' 'webkb.gz' "$DIR"

download_and_unzip 'ana' 'https://www.dropbox.com/s/p0xr0oe6sffb4xi/phd-datasets.zip?dl=0#' 'phd-datasets.zip' "$DIR"

download_and_unzip 'tagmynews' 'http://acube.di.unipi.it/repo/news.gz' 'news.gz' "$DIR"

download_and_unzip 'rotten_imdb' 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz' 'rotten_imdb.tar.gz' "$DIR"

download_and_unzip 'review_polarity' 'http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz' 'review_polarity.tar.gz' "$DIR"

download_and_unzip 'ling-spam' 'http://csmining.org/index.php/ling-spam-datasets.html?file=tl_files/Project_Datasets/Ling-Spam%20data%20set/lingspam_public.tar.tar' 'lingspam_public.tar.tar' "$DIR"