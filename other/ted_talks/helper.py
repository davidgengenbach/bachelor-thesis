import re
import requests
import os
import pandas as pd
import collections
from glob import glob
import pickle

HTML_FOLDER = 'data/html'
URL_TRANSCRIPTS = 'https://www.kaggle.com/rounakbanik/ted-talks/downloads/transcripts.csv'

TAG_REGEXP = r"<meta property=\"og:video:tag\" content=\"(.+?)\" \/\>"
def get_tags_from_html(html):
    matches = re.findall(TAG_REGEXP, html)
    if not len(matches):
        return None
    tags = [x.lower().strip() for x in matches]
    return tags


def clean_for_filename(text):
    return "".join(x for x in text if x.isalnum())


def get_transcripts(cache_file='data/transcripts.csv'):
	folder = cache_file.rsplit('/', 1)[0]
	os.makedirs(folder, exist_ok=True)

	if not os.path.exists(cache_file):
		raise Exception('Transcript CSV file "{}" does not exist! Please download it from {}'.format(cache_file, URL_TRANSCRIPTS))

	df = pd.read_csv(cache_file)

	def get_word_count(t):
	    return len(t.split(' '))

	df['word_count'] = df.transcript.apply(get_word_count)
	df['url'] = df.url.str.strip()
	df['url_clean'] = df.url.apply(clean_for_filename)
	df.url.drop_duplicates(inplace=True)
	return df


def get_html_data(folder = HTML_FOLDER, cache_file='data/html.npy'):
	if os.path.exists(cache_file):
		with open(cache_file, 'rb') as f:
			return pickle.load(f)

	if not os.path.exists(folder):
		raise Exception('HTML folder "{}" does not exist. Aborting. Please run the "download_ted_talks.py" script first!'.format(folder))
	html_files = glob('{}/*.txt'.format(folder))

	if not len(html_files):
		raise Exception('No html files found in: "{}"'.format(folder))

	html_data = {}
	for file in html_files:
		with open(file) as f:
			html = f.read()
		url, html = [x.strip() for x in html.split('\n\n', 1)]
		if url in html_data: continue
		html_data[url] = html.strip()

	with open(cache_file, 'wb') as f:
		pickle.dump(html_data, f)

	return html_data


def add_talk_tags(df):
	html_data = get_html_data()
	data = collections.defaultdict(list)
	for url, html in html_data.items():
		data['url'].append(url)
		tags = get_tags_from_html(html)
		data['tags'].append(tags)
		data['num_tags'].append(len(tags))
	return df.merge(right = pd.DataFrame(data), on = 'url', validate = 'one_to_one')


def get_transcripts_with_tags_df():
	return add_talk_tags(get_transcripts())