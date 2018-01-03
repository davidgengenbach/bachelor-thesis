import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(dir_path, '..')
sys.path.append(parent_dir)
os.chdir(parent_dir)

from glob import glob
from IPython.display import display
from PIL import Image
from preprocessing import preprocessing
from sklearn import dummy
from sklearn import metrics, pipeline, preprocessing, svm
import transformers
from transformers.pipelines import *
from utils import results_helper, filename_utils, helper, graph_helper, helper, dataset_helper, cooccurrence
import collections
import copy
import functools
import gc
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import seaborn as sns
import shutil
import sklearn
import sympy
import sys
import unicodedata, re
import warnings
import psutil
import matplotlib.style as style
from pprint import pprint
from utils.graph_helper import *
import experiments
from experiments import experiment_helper
from itertools import chain
from utils import significance_test_utils

import tqdm

def log_progress_nb(*args):
    return tqdm.tqdm_notebook(*args)


EXPORT_DPI = 100
EXPORT_FIG_SIZE = (8, 4)
EXPORT_FIG_SIZE_BIG = (10, 7)
EXPORT_FIG_WIDTH, EXPORT_FIG_HEIGHT = EXPORT_FIG_SIZE
EXPORT_FIG_WIDTH_BIG, EXPORT_FIG_HEIGHT_BIG = EXPORT_FIG_SIZE_BIG

pd.set_option('precision', 4)
pd.options.display.max_rows = 80
pd.options.display.max_columns = 999
pd.options.display.max_colwidth = -1

sns.set('paper', 'whitegrid', palette = 'deep')
params = [
    ('axes.titleweight', 'normal'),
    ('axes.titlepad', 10),
    ('axes.titlesize', 11),
    ('axes.labelweight', 'normal'),
    ('axes.labelsize', 9),
    ('figure.titlesize', 13),
    ('figure.titleweight', 'normal'),
    ('figure.figsize', EXPORT_FIG_SIZE_BIG),
    ('figure.dpi', EXPORT_DPI),
    ('font.family', 'sans-serif'),
    ('font.sans-serif', ['Verdana'])
]

for name, val in params:
    plt.rcParams[name] = val


def cleanup_axes(ax, remove_splines=True):
    ax.grid('off')
    if remove_splines:
        for pos, spine in ax.spines.items(): spine.set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


IMAGE_FOLDER = 'tmp'
os.makedirs(IMAGE_FOLDER, exist_ok=True)

def save_fig(fig, filename_without_ext, folder = IMAGE_FOLDER, extensions = ['png', 'pdf'], dpi=72):
    os.makedirs(folder, exist_ok=True)
    for ext in extensions:
        filename = '{}/{}.{}'.format(folder, filename_without_ext, ext)
        fig.savefig(filename, dpi=dpi)


DATASETS = dataset_helper.get_all_available_dataset_names()
for dataset in DATASETS:
    globals()['DATASET_' + dataset.upper().replace('-', '_')] = dataset

TASK_TYPES = experiments.get_all_task_types()
for task_type in TASK_TYPES:
    globals()['TASK_TYPE_{}'.format(task_type.upper())] = task_type

def get_text_file_lines(file, remove_empty = True, strip=True):
    with open(file) as f:
        return [x.strip() if strip else x for x in f if not remove_empty or x.strip() != '']

def get_pickle(f):
    with open(f, 'rb') as f:
        return pickle.load(f)