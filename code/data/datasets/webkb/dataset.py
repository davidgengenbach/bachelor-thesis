import os
from glob import glob
from bs4 import BeautifulSoup
import codecs
import io
import pickle
import sys
from joblib import Parallel, delayed

PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))

def fetch(use_cached = True, n_jobs = 2):
    npy_file = os.path.join(PATH_TO_HERE, 'src', 'docs.npy')

    if use_cached and os.path.isfile(npy_file):
        with open(npy_file, 'rb') as f:
            return pickle.load(f)

    folder = os.path.join(PATH_TO_HERE, 'src', 'webkb')
    categories = get_categories(folder)
    data = []

    for idx, cat in enumerate(categories):
        sys.stdout.write('\rCategory: {:>2}/{}'.format(idx + 1, len(categories)))
        cat_folder = os.path.join(folder, cat)
        files = glob(cat_folder + '/*/*')
        data += Parallel(n_jobs=n_jobs)(delayed(get_text_from_cat)(cat, d) for d in files)
    print('\rCompleted reading in categories')
    print()
    data = [x for x in data if x[0]]
    with open(npy_file, 'wb') as f:
        pickle.dump(data, f)
    return data

def get_text_from_cat(cat, file):
    with io.open(file, errors = 'ignore') as f:
        try:
            text = f.read()
            text = "\n".join(text.split('\n\n', maxsplit=1)[1:])
            return (cat, get_text_from_html(text))
        except Exception as e:
            print("Read error: {}\nException: {}\n".format(file, e))
    return (None, None)

def get_categories(folder):
    return [x.strip() for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x))]

def get_text_from_html(html):
    parsed = BeautifulSoup(html, 'lxml')
    html = parsed.find('html')
    return html.text if html else parsed.get_text()

if __name__ == '__main__': fetch()


#if idx % 10 == 0: sys.stdout.write('\r\t{:>3}/{}'.format(idx, len(files) - 1))