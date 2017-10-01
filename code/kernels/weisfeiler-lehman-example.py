
# coding: utf-8


import collections
import copy
import functools
import os
import pickle
from glob import glob
from time import time

import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import psutil
import sklearn
from IPython.display import display
from joblib import Parallel, delayed
from matplotlib import cm
from scipy.sparse import lil_matrix, csr_matrix, vstack
from sklearn import linear_model
from sklearn import utils
from wl import *

from utils import helper, dataset_helper


def flatten(l, as_set = False):
    return functools.reduce(lambda acc, x: acc | set(x) if as_set else acc + list(x), l, set() if as_set else list())


# ## Retrieve graphs from Tobias' concept-graph extraction library


DATASET = 'ling-spam'
#cache_file = dataset_helper.CACHE_PATH + '/dataset_graph_gml_{}-single.npy'.format(DATASET)
cache_file = dataset_helper.CACHE_PATH + '/dataset_graph_gml_{}-single.npy'.format(DATASET)
X, Y = dataset_helper.get_dataset(DATASET, use_cached = True, cache_file = cache_file)
#X, Y = dataset_helper.get_subset_with_most_frequent_classes(X, Y, num_classes_to_keep = 3)
assert len(X) and len(Y), 'Dataset is empty: {}'.format(DATASET)
assert len(X) == len(Y)
graphs_per_topic = dataset_helper.get_dataset_dict(X, Y)
#graphs_per_topic = get_graphs(GRAPH_DIR, undirected = False, verbose = False, limit_elements = -1)


# ## Statistics about the graphs


df_graphs_per_topic = pd.DataFrame([(topic, len(graphs), [len(x.nodes()) for x in graphs], [len(x.edges()) for x in graphs]) for topic, graphs in graphs_per_topic.items()], columns = ['topic', 'num_graphs', 'num_nodes', 'num_edges']).set_index(['topic']).sort_values(by = 'num_graphs')
ax = df_graphs_per_topic.plot.barh(title = 'Graphs per topic', legend = False, figsize = (14, 8))
ax.set_xlabel('# graphs')
plt.show()

def get_range_of(df, column):
    return df[column].apply(lambda x: min(x)).min(), df[column].apply(lambda x: max(x)).max()

nodes_range = get_range_of(df_graphs_per_topic, 'num_nodes')
edges_range = get_range_of(df_graphs_per_topic, 'num_edges')

if False:
    num_classes = len(set(Y))
    ncols, nrows = (2, max(2, math.ceil(num_classes / 2)))
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize = (14, 20))
    fig_, axes_ = plt.subplots(ncols=ncols, nrows=nrows, figsize = (14, 20))
    for idx, (topic, (num_graphs, num_nodes, num_edges)) in enumerate(df_graphs_per_topic.iterrows()):
        row = int(idx / ncols)
        col = idx % ncols
        print(row, col)
        ax = axes[row][col]
        ax_ = axes_[row][col]

        # 
        ax.set_title("{}".format(topic))
        ax.set_xlabel('# nodes')
        ax.hist(num_nodes, bins=40, normed = True, range = nodes_range)

        #
        ax_.set_title("{}".format(topic))
        ax_.set_xlabel('# edges')
        ax_.hist(num_edges, bins=40, normed = True, range = edges_range)
    fig.tight_layout()
    fig_.tight_layout()
plt.show()



df_graphs_per_topic['avg_nodes'] = df_graphs_per_topic.num_nodes.apply(lambda x: np.mean(x))
df_graphs_per_topic['avg_edges'] = df_graphs_per_topic.num_edges.apply(lambda x: np.mean(x))
df_graphs_per_topic.plot(kind = 'barh', y = ['avg_nodes', 'avg_edges'], figsize = (14, 8))
plt.show()


# ## Filter categories


REMOVE_CATEGORIES = ['misc.forsale', 'comp.graphics']
REMOVE_CATEGORIES = []
graphs = graphs_per_topic.copy()
if len(REMOVE_CATEGORIES):
    for cat in REMOVE_CATEGORIES:
        del graphs[cat]



all_graphs = flatten(graphs.values())



all_nodes = set()
for g in all_graphs:
    all_nodes |= set(g.nodes())



print('#graphs:\t\t{}'.format(len(all_graphs)))
print('Unique tokens found:\t{}'.format(len(all_nodes)))


# ## Create train/test split


# TODO: This should be done with dataset_helper.split_dataset(..)
def get_train_test_split(topic_graphs, train_split_ratio = 0.8):
    train = []
    test = []
    num_elements = {}
    for topic, graphs in topic_graphs.items():
        num_elements_train = int(len(graphs) * train_split_ratio)
        train += [(topic, x) for x in graphs[:num_elements_train]]
        test += [(topic, x) for x in graphs[num_elements_train:]]
    return train, test
train, test = get_train_test_split(graphs)



print('#graphs\n\ttrain:\t{}\n\ttest:\t{}'.format(len(train), len(test)))



for set_ in [train, test]:
    X, Y = [x for label, x in set_], [label for label, x in set_]
    dataset_helper.plot_dataset_class_distribution(X, Y)
plt.show()



H = 2


# ## Calculate phi and gram-matrix of WL kernel for training instances


for set_ in [train, test]:
    for topic, graph in set_:
        if nx.number_of_edges(graph) == 0: set_.remove((topic, graph))




# ### Train classifier with the phi s of the training instances


clf = sklearn.linear_model.Perceptron(n_iter = 500, verbose = False, n_jobs = -1, class_weight='balanced')
X = phi_list_train[-1].T
Y = [topic for topic, graphs in train]
clf.fit(X, Y)


# ## Get predictions for test instances
# 
# ### Calculate phi for each test instance




nodes_nums = sum([nx.number_of_nodes(graph) for topic, graph in train])
nodes_nums


# ### Predict test instances



# ### Metrics


Y_real = [topic for topic,graph in USED_SET]
Y_pred = predicted






# #### About sparsity of test phi


df_phi_test_non_zero_elements = pd.DataFrame(list(zip(Y_real, Y_pred, [np.count_nonzero(x) for x in phi_test])), columns = ['real_topic', 'pred_topic', 'num_phi_non_zero'])



display(df_phi_test_non_zero_elements.groupby(by = 'real_topic').describe())
display(df_phi_test_non_zero_elements.groupby(by = 'pred_topic').describe())



max_ = 0
vals_over_1 = 0
for i in phi_test:
    vals = i[i > 1]
    if not len(vals): continue
    vals_over_1 += 1
    m = max(vals)
    if m > max_:
        max_ = m
        print(max_)
print(vals_over_1)



cf_mat = sklearn.metrics.confusion_matrix(Y_real, Y_pred)
classes = min(len(set(Y_real)) * 2, 30)
fig = plt.figure(figsize=(classes, classes))
helper.plot_confusion_matrix(cf_mat, clf.classes_, normalize = True)
plt.show()



sklearn.metrics.f1_score(Y_real, Y_pred, average='macro')

