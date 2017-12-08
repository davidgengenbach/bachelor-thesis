#!/usr/bin/env python3

import csv
import numpy as np
import pandas as pd
from glob import glob
import os, sys
from bs4 import BeautifulSoup
import sys
import re

import pickle

file = 'meta.csv'
header = ['filename', 'title', 'print_section', 'desk', 'online_sections', 'word_count']

with open(file) as f:
    reader = csv.DictReader(f, fieldnames=header)
    rows = list(reader)

df = pd.DataFrame(rows).set_index('filename')
for c in df:
    df[df[c] == '_'] = np.nan
df['word_count'] = pd.to_numeric(df.word_count)

def get_articles_with_word_counts(df, low, high):
    return df[(df.word_count > low) & (df.word_count < high)]

df_filtered = get_articles_with_word_counts(df, 3000, df.word_count.quantile(0.9999))


f = df_filtered[df_filtered.online_sections.str.contains(';') == False].online_sections.value_counts().to_frame()
filtered_online_section = f[f.online_sections > 250].index.values
df_filtered_filtered = df_filtered[df_filtered.online_sections.apply(lambda x: x in filtered_online_section)]



prefix = 'filtered_articles/'
filtered_files = glob('{}*/*/*/*.xml'.format(prefix))
filtered_files_ = ['/'.join(x.rsplit('/', 4)[-4:]) for x in filtered_files]
# Test whether all articles are there
assert len(filtered_files_) == len(set(filtered_files_) & set(df_filtered_filtered.index.values))


def get_body_of_article(file):
    assert os.path.exists(file)
    with open(file) as f:
        content = f.read()
    body = re.findall(r'<block class="full_text">(.+?)</block>', content, re.DOTALL | re.MULTILINE)
    assert len(body) == 1
    body = body[0].strip().replace('<p>', '').replace('</p>', '')
    return body

bodies = {}
for idx, file in enumerate(filtered_files):
    sys.stdout.write('\r{:9}/{}'.format(idx + 1, len(filtered_files)))
    body = get_body_of_article(file)
    bodies[file.replace(prefix, '')] = body

bodies_sorted = []
for filename, df_ in df_filtered_filtered.iterrows():
    assert filename in bodies
    bodies_sorted.append(bodies[filename])

df_filtered_filtered['body'] = bodies_sorted


X = df_filtered_filtered.body.values
Y = df_filtered_filtered.online_sections.values

with open('dataset_nyt.npy', 'wb') as f:
    pickle.dump((X, Y), f)