"""TagMyNews NEWS dataset

Link
    http://acube.di.unipi.it/tmn-dataset/

Download
    http://acube.di.unipi.it/repo/news.gz
"""

import codecs
import os
import pandas as pd

current_folder = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.join(current_folder, 'src', 'news.txt')


def fetch(filename=FILENAME):
    title_fields = ['title', 'description', 'link', 'id', 'date', 'source', 'category']

    with codecs.open(filename) as f:
        text = f.read().strip()

    records = [x.strip().split('\n') for x in text.split('\n\n') if x.strip().count('\n') == len(title_fields) - 1]

    df = pd.DataFrame(records, columns=title_fields).set_index('id')
    df['date'] = pd.to_datetime(df.date)

    def record_to_text(record):
        return '{record.description}'.format(record=record)
        # return '{record.title} {record.description}'.format(record=record)

    X, Y = zip(*[(record_to_text(item), item.category) for idx, item in df.iterrows()])
    return list(X), list(Y)
