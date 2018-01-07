#!/usr/bin/env bash

# Exit on error
set -e

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

sep() {
    echo
    echo "------------------------------------------------------------------------------"
    echo
}

sep
echo "## This script will bootstrap the project by: "
echo "- Installing the needed Python dependencies"
echo "- Downloading language models"
echo "- Downloading the datasets"

sep
echo "## Installing Python packages"
echo "Note that the code has only been tested with Python 3.6."
echo "We recommend using anaconda: https://www.anaconda.com/download/#linux"
echo "If this fails, try installing the build-essentials for your distribution"
pip3 install -r requirements.txt > /dev/null

sep
echo "## Downloading language model (en) and nltk stopwords"
python3 -m spacy download en > /dev/null
python3 -c 'import nltk; nltk.download("stopwords")'
python3 -c 'import nltk; nltk.download("wordnet")'

sep
echo "## Downloading datasets"

cd data/datasets
DATASET_DIR="$(pwd -P)"

download_and_unzip() {
    NAME="$1"
    LINK="$2"
    FILE="$3"
    BACK_DIR="$4"

    echo -e "\n\tDownloading: $NAME"
    echo

    folder="$NAME/src"
    mkdir -p $folder
    cd $folder

    wget -q --show-progress --continue -O "$FILE" "$LINK" > /dev/null || true
    tar -zxvf "$FILE" || gunzip -d "$FILE" || unzip "$FILE" || echo "Warning: could not unzip/gunzip or un-tar $FILE"
    cd "$BACK_DIR"
}

download_and_unzip 'webkb' 'http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/webkb-data.gtar.gz' 'webkb.gz' "$DATASET_DIR"

download_and_unzip 'ana' 'https://www.dropbox.com/s/p0xr0oe6sffb4xi/phd-datasets.zip?dl=0#' 'phd-datasets.zip' "$DATASET_DIR"

download_and_unzip 'tagmynews' 'http://acube.di.unipi.it/repo/news.gz' 'news.gz' "$DATASET_DIR"

download_and_unzip 'rotten_imdb' 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz' 'rotten_imdb.tar.gz' "$DATASET_DIR"

download_and_unzip 'review_polarity' 'http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz' 'review_polarity.tar.gz' "$DATASET_DIR"

download_and_unzip 'ling-spam' 'http://csmining.org/index.php/ling-spam-datasets.html?file=tl_files/Project_Datasets/Ling-Spam%20data%20set/lingspam_public.tar.tar' 'lingspam_public.tar.tar' "$DATASET_DIR"