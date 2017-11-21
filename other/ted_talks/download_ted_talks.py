#!/usr/bin/env python3

import pandas as pd
import requests
import re
import sys
import time
import os
import logging


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('Crawler')
logger.setLevel(logging.DEBUG)

df = pd.read_csv('data/transcripts.csv')

def get_word_count(t):
    return len(t.split(' '))

df['mean_word_count'] = df.transcript.apply(get_word_count)

#df = df[(df.mean_word_count > 1500) & (df.mean_word_count < 4000)]

WAIT_AFTER_RES = 10
WAIT_AFTER_BAN = 40
HTML_FOLDER = 'data/html'
os.makedirs(HTML_FOLDER, exist_ok=True)

HTML_FILE_TPL = '{}/{}.html'.format(HTML_FOLDER, '{}')

TAG_REGEXP = r"\"tags\":\[(.+?)\]"

def get_filename_from(df):
    return HTML_FILE_TPL.format("".join(x for x in df.url if x.isalnum()))

def get_html(url):
    res = requests.get(url)
    if res.status_code != 200:
        return res
    return res.text

c = 0
num_all = len(df)
for i, (idx, df_) in enumerate(df.iterrows()):
    url = df_.url.strip()
    if os.path.exists(get_filename_from(df_)): continue

    start = time.time()

    try:
        html = get_html(url)
        
        if isinstance(html, requests.Response):
            logger.warning('Too many requests, waiting {}s'.format(WAIT_AFTER_BAN))
            time.sleep(WAIT_AFTER_BAN)
            continue
        
        assert html and isinstance(html, str)
        
        with open(get_filename_from(df_), 'w') as f:
            html = '{}\n\n{}'.format(df_.url, html)
            f.write(html)

    except Exceptions as e:
        logger.warning('Error: {}'.format(e))
    logger.info('({:4}/{}): {:.2f}s'.format(i, len(df), time.time() - start))
    time.sleep(WAIT_AFTER_RES)