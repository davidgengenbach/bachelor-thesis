import matplotlib.pyplot as plt
import numpy as np
import itertools
import codecs
import functools
from glob import glob
import sklearn
from sklearn.datasets import base

DATASET_FOLDER = 'datasets'

CACHE = {}


def get_all_datasetnames(dataset_folder = DATASET_FOLDER):
    return set([filename.split('/')[1].split('-')[0] for filename in glob(r'{}/*.txt'.format(DATASET_FOLDER))])


def _get_data_container():
    data_container = base.Bunch()
    data_container.target_names = []
    data_container.data = []
    return data_container

def get_dataset(dataset, subset, typ = 'stemmed', category_filter = None, dataset_folder = DATASET_FOLDER):
    if dataset not in CACHE:
        CACHE[dataset] = {}
    if typ not in CACHE[dataset] or subset not in CACHE[dataset][typ]:
        with codecs.open('{}/{}-{}-{}.txt'.format(dataset_folder, dataset, subset, typ), encoding = 'utf-8') as f:
            els = f.read().split('\n')
            def red_fn(acc, x):
                if x.strip() == '' or len(x.strip().split('\t', 1)) == 1:
                    return acc
                target, words = x.strip().split('\t', 1)
                acc.target_names.append(target)
                acc.data.append(words)
                return acc
            data = functools.reduce(red_fn, els, _get_data_container())
            if typ not in CACHE[dataset]:
                CACHE[dataset][typ] = {subset: data}
            else:
                CACHE[dataset][typ][subset] = data
    SET = CACHE[dataset][typ][subset]
    add_target_classes(SET)
    if category_filter:
        filtered = _get_data_container()
        for idx, (target_name, words) in enumerate(zip(SET.target_names, SET.data)):
            if target_name in category_filter:
                filtered.target_names.append(target_name)
                filtered.data.append(words)
        return filtered
    else:
        return SET

def add_target_classes(data_container):
    if 'target' in data_container:
        return
    uniq_target_names = list(set(data_container.target_names))
    target_name_2_id = {target_name: idx for idx, target_name in enumerate(uniq_target_names)}
    data_container.target = []
    for target_name in data_container.target_names:
        data_container.target.append(target_name_2_id[target_name])

    data_container.target_names = uniq_target_names

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          round_confusion = 2):
    """
    Plots the confusion matrix.
    Taken from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    print(classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

        
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = int(round(cm[i, j], round_confusion) * 100) if round_confusion else cm[i, j]
        val = '{}%'.format(val)
        plt.text(j, i, val,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def log_progress(sequence, every=None, size=None, name='Items'):
    """
    https://github.com/alexanderkuk/log-progress
    """
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )