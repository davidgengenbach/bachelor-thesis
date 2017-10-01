"""Subjectivity dataset (rotten, imdb)

Paper
    https://arxiv.org/pdf/cs/0409058.pdf
    
Link
    http://www.cs.cornell.edu/people/pabo/movie-review-data/

Download
    http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz
"""


import codecs
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
SRC_FOLDER = os.path.join(current_folder, 'src')

def fetch(src_folder=SRC_FOLDER, file_quote = 'quote.tok.gt9.5000', file_plot = 'plot.tok.gt9.5000'):
    X, Y = [], []

    records = []
    for file_name, class_name, encoding in zip([file_quote, file_plot], ['subjective', 'objective'], ['windows-1252', 'utf-8']):
        with codecs.open(os.path.join(src_folder, file_name), encoding = encoding) as f:
            text = f.read().strip()

        for line in text.split('\n'):
            line = line.strip()
            if line == '': continue
            
            records.append((line, class_name))
    X, Y = zip(*records)
    return list(X), list(Y)
