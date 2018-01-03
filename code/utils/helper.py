# If no $DISPLAY is given, this fails
try:
    import matplotlib.pyplot as plt
except:
    pass

import psutil
import numpy as np
import itertools
import os
import argparse
from utils import time_utils, git_utils, constants
import random

DATASET_FOLDER = 'data/datasets'

def flatten_array(arr) -> list:
    flattened = []
    for x in arr:
        flattened += list(x)
    return flattened

def print_script_args_and_info(args):
    print('Starting:\n\t{}'.format(time_utils.get_time_formatted()))
    print()

    print('Git commit:\n\t{}'.format(git_utils.get_current_commit()))
    print()

    print('Arguments:\t')
    for key, val in vars(args).items():
        print('\t{:30} {}'.format(key, val))
    print()

def argparse_str2bool(v):
    '''
    From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def plot_confusion_matrix(cm,
                          classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          round_confusion=2,
                          x_rotation=90,
                          show_non_horizontal_percent=True):
    import plt
    """
    Plots the confusion matrix.
    Taken from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    cmap = plt.cm.Blues
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=x_rotation)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if not show_non_horizontal_percent and i != j:
            continue
        val = int(round(cm[i, j], round_confusion) * 100) if round_confusion else cm[i, j]
        val = '{}%'.format(val)
        plt.text(j, i, val,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_human_readable_size(num):
    if not isinstance(num, int): return
    exp_str = [ (0, 'B'), (10, 'KB'),(20, 'MB'),(30, 'GB'),(40, 'TB'), (50, 'PB'),]               
    i = 0
    while i+1 < len(exp_str) and num >= (2 ** exp_str[i+1][0]):
        i += 1
        rounded_val = round(float(num) / 2 ** exp_str[i][0], 2)
    return '%s %s' % (int(rounded_val), exp_str[i][1])


def get_memory_footprint():
    process = psutil.Process(os.getpid())
    #memory = psutil.virtual_memory()
    #mem = {k: get_human_readable_size(getattr(memory, k)) for k in dir(memory) if not k.startswith('_') and isinstance(getattr(memory, k), int)}
    return get_human_readable_size(process.memory_info().rss)


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

def print_timestamps(t, width=30, only_total=False):
    from utils import time_utils
    timestamps = sorted(t.items(), key=lambda x: x[1])
    print('Total: {}'.format(time_utils.seconds_to_human_readable(timestamps[-1][1] - timestamps[0][1])))
    if only_total: return
    for idx, (name, time_) in enumerate(timestamps[:-1]):
        next_time = timestamps[idx + 1][1]
        print(name.center(width))
        print('|'.center(width))
        print(time_utils.seconds_to_human_readable(next_time - time_).center(width))
        print('|'.center(width))
    print(timestamps[-1][0].center(width))

def set_random_seed(seed=None):
    if seed is None:
        seed = constants.RANDOM_SEED
    np.random.seed(seed)
    random.seed(seed)

set_random_seed()


