from glob import glob
import sys
from bs4 import BeautifulSoup
import mechanicalsoup
import collections
import os
import pandas as pd
import pickle
import json

SCRIPTS_JSON="scripts.json"
HTML_FOLDER='data/html/'
SCRIPTS_PICKLE='data/scripts.pkl'
START_URL = 'http://www.imsdb.com'
RSS_GENRE_LINK = 'http://www.imsdb.com/feeds/genre.php?genre={}'
SCRIPTS_PARSED_FOLDER = 'data/processed/'
SCRIPTS_PARSED_PICKLE = 'data/scripts_parsed.pkl'
LINK_2_LABEL_FILE = 'data/link_2_newlabel.json'

def get_scripts_df(cache_file=SCRIPTS_JSON, use_cache=True):
    if use_cache and os.path.exists(cache_file):
        return pd.read_json(SCRIPTS_JSON)

    df_genres = get_genres()
    data = []
    for genre, df_ in df_genres.groupby('genre'):
        url = RSS_GENRE_LINK.format(genre).lower()
        result = feedparser.parse(url)
        entries = result['entries']
        for entry in entries:
            entry['genre'] = genre
            data.append(entry)

    df_scripts = pd.DataFrame(data)
    df_scripts.to_json(cache_file)
    return df_scripts

def get_genres():
    b = mechanicalsoup.Browser()
    page = b.get(START_URL)
    links_alphabetical = page.soup.select('a[href^="/alphabetical"]')
    links_genres = page.soup.select('a[href^="/genre"]')
    data = collections.defaultdict(lambda: [])
    for link in links_genres:
        url = START_URL + link.attrs['href']
        genre = link.text
        print('Retrieving genre: {}'.format(genre))
        data['genre'].append(genre)
        data['url'].append(url)
    df_genres = pd.DataFrame(data)
    return df_genres

def get_scripts(folder=HTML_FOLDER):
    files = glob('{}/*html'.format(folder))
    scripts = []
    for idx, file in enumerate(files):
        sys.stdout.write('\r{:4}/{}'.format(idx + 1, len(files)))
        with open(file) as f:
            scripts.append(f.read().split('\n\n', 1))
    return scripts

def extract_scripts(scripts):
    extracted = []
    for idx, (link, script_html) in enumerate(scripts):
        sys.stdout.write('\r{:4}/{}'.format(idx + 1, len(scripts)))
        content = BeautifulSoup(script_html, 'lxml').select('.scrtext > pre')
        if not len(content): continue
        assert len(content) == 1
        extracted.append((link, content[0]))
    return extracted

def get_extracted_scripts(use_cache=True, cache_file=SCRIPTS_PICKLE):
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    scripts = get_scripts()
    scripts_ = extract_scripts(scripts)
    scripts_ = [(link, str(content)) for link, content in scripts_]

    with open(cache_file, 'wb') as f:
        pickle.dump(scripts_, f)

    return get_extracted_scripts(use_cache=True)

def clean_filename(f):
    return "".join(x for x in f if x.isalnum())

def save_scripts(scripts, folder=SCRIPTS_PARSED_FOLDER, cache_file=SCRIPTS_PARSED_PICKLE):
    os.makedirs(folder, exist_ok=True)
    for link, script in scripts:
        filename = folder + clean_filename(link) + '.txt'
        with open(filename, 'w') as f:
            f.write('{}\n\n{}'.format(link, script))
    with open(cache_file, 'wb') as f:
        pickle.dump(scripts, f)

def get_processed_scripts(use_cache=True, folder=SCRIPTS_PARSED_FOLDER, cache_file=SCRIPTS_PARSED_PICKLE):
    if use_cache and os.path.exists(SCRIPTS_PARSED_PICKLE):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    scripts = []
    for file in glob('{}/*.txt'.format(folder)):
        with open(file) as f:
            script = f.read().split('\n\n', 1)
            scripts.append(script)
    return scripts

def get_link_2_label(cache_file=LINK_2_LABEL_FILE):
    with open(cache_file) as f:
        return json.load(f)