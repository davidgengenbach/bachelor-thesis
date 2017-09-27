"""Polarity dataset

Paper
    http://www.cs.cornell.edu/home/llee/papers/cutsent.pdf
    
Link
    http://www.cs.cornell.edu/people/pabo/movie-review-data/

Download
    http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz
"""


import codecs
import os
from glob import glob

current_folder = os.path.dirname(os.path.abspath(__file__))
SRC_FOLDER = os.path.join(current_folder, 'src', 'review_polarity', 'txt_sentoken')

def fetch(src_folder=SRC_FOLDER):
    X, Y = [], []
    classes = ['neg', 'pos']

    records = []
    for clazz in classes:
        for file in glob(os.path.join(src_folder, clazz, '*.txt')):
            with codecs.open(file) as f:
                text = f.read().strip()
            record = (text, clazz)
            records.append(record)
    X, Y = zip(*records)
    return list(X), list(Y)
