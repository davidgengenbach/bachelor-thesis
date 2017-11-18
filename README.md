# Text Classification using Concept Maps

[Protocol](other/protocol.md)

## Install Instructions

```shell
cd code
pip install -r requirements.txt
./script_download_datasets.sh
python -m spacy download en
python -c 'import nltk; nltk.download("stopwords")'
```