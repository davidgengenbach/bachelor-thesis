
from glob import glob
from glob import glob 
from IPython.display import display
from PIL import Image
from preprocessing import preprocessing
from sklearn import dummy
from sklearn import metrics, pipeline, preprocessing, svm
from transformers import fast_wl_pipeline, text_pipeline
from transformers.fast_wl_graph_kernel_transformer import FastWLGraphKernelTransformer
from transformers.nx_graph_to_tuple_transformer import NxGraphToTupleTransformer
from transformers.tuple_selector import TupleSelector
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


EXPORT_DPI = 100
EXPORT_FIG_SIZE = (8, 4)
EXPORT_FIG_SIZE_BIG = (10, 7)
EXPORT_FIG_WIDTH, EXPORT_FIG_HEIGHT = EXPORT_FIG_SIZE
EXPORT_FIG_WIDTH_BIG, EXPORT_FIG_HEIGHT_BIG = EXPORT_FIG_SIZE_BIG

pd.options.display.max_rows = 80
pd.options.display.max_columns = 999
pd.options.display.max_colwidth = -1

sns.set('notebook', 'whitegrid', palette = 'deep')
plt.rcParams['figure.figsize'] = EXPORT_FIG_SIZE_BIG
plt.rcParams['figure.dpi'] = EXPORT_DPI


