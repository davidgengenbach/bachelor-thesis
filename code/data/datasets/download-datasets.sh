#!/usr/bin/env bash

# https://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

download_and_unzip() {
    NAME="$1"
    LINK="$2"
    FILE="$3"
    BACK_DIR="$4"

    folder="$NAME/src"
    mkdir -p $folder
    cd $folder

    wget --continue -O "$FILE" "$LINK" || true
    tar -zxvf "$FILE" || gunzip -d "$FILE" || unzip "$FILE"
    cd "$BACK_DIR"
}

# TODO: reuters-21578
# TODO: webkb

# ana
download_and_unzip 'ana' 'https://www.dropbox.com/s/p0xr0oe6sffb4xi/phd-datasets.zip?dl=0#' 'phd-datasets.zip' "$DIR"

# tagmynews
download_and_unzip 'tagmynews' 'http://acube.di.unipi.it/repo/news.gz' 'news.gz' "$DIR"

# rotten_imdb
download_and_unzip 'rotten_imdb' 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz' 'rotten_imdb.tar.gz' "$DIR"

# review_polarity
download_and_unzip 'review_polarity' 'http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz' 'review_polarity.tar.gz' "$DIR"

# ling-spam
download_and_unzip 'ling-spam' 'http://csmining.org/index.php/ling-spam-datasets.html?file=tl_files/Project_Datasets/Ling-Spam%20data%20set/lingspam_public.tar.tar' 'lingspam_public.tar.tar' "$DIR"

# I do not know what these are for, the homepage does not describe the difference

#download_and_unzip 'ling-spam' 'http://csmining.org/index.php/ling-spam-datasets.html?file=tl_files/Project_Datasets/Ling-Spam%20data%20set/lingspam_public01.tar.tar' 'lingspam_public01.tar.tar' "$DIR"
#download_and_unzip 'ling-spam' 'http://csmining.org/index.php/ling-spam-datasets.html?file=tl_files/Project_Datasets/Ling-Spam%20data%20set/lingspam_public02.tar.tar' 'lingspam_public02.tar.tar' "$DIR"
