import os
from glob import glob
from bs4 import BeautifulSoup
import codecs
import io
import pickle
import sys
from joblib import Parallel, delayed

PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(n_jobs=2):
    folder = os.path.join(PATH_TO_HERE, 'src', 'webkb')
    categories = get_categories(folder)
    X, Y = [], []
    for idx, cat in enumerate(categories):
        sys.stdout.write('\rCategory: {:>2}/{}'.format(idx + 1, len(categories)))
        cat_folder = os.path.join(folder, cat)
        files = glob(cat_folder + '/*/*')
        new_data = Parallel(n_jobs=n_jobs)(delayed(get_text_from_cat)(cat, d) for d in files)
        for x, y in new_data:
            if not x or not y:
                continue
            X.append(x)
            Y.append(y)
    print('\rCompleted reading in categories')
    print()
    return X, Y


def get_text_from_cat(cat, file):
    with io.open(file, errors='ignore') as f:
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

if __name__ == '__main__':
    fetch()
