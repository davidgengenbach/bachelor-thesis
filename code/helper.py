import matplotlib.pyplot as plt
import numpy as np
import itertools
import codecs
import functools
from glob import glob
import sklearn
from sklearn.datasets import base

DATASET_FOLDER = 'data/datasets'

CACHE = {}


class MostFrequentLabelClassifier(object):
     
    def __init__(self):
        pass

    def fit(self, x = None, y = None):
        label_counts = {}
        for y_ in y:
            if y_ not in label_counts:
                label_counts[y_] = 0
            label_counts[y_] += 1
        self.label_to_return = max(label_counts.items(), key = lambda x: x[1])[0]

    def predict(self, x = None):
        return [self.label_to_return] * x.shape[0]

def get_classifiers(iterations = 500):
    return {
        'PassiveAggressiveClassifier': sklearn.linear_model.PassiveAggressiveClassifier(),
        'Perceptron': sklearn.linear_model.Perceptron(n_iter = iterations),
        'LogisticRegression': sklearn.linear_model.LogisticRegression(max_iter = iterations),
        'SGDClassifier': sklearn.linear_model.SGDClassifier(n_iter = iterations),
        'MostFrequentLabel': MostFrequentLabelClassifier()
    }


def get_tsne_embedding(model_w2v):
    w2v_vectors = model_w2v[model_w2v.wv.vocab]
    tsne = sklearn.manifold.TSNE(n_components=2)
    tsne_vectors = tsne.fit_transform(w2v_vectors)
    return tsne_vectors

def plot_embedding(model_w2v):
    indexed_vocab = {v.index: k for k, v in model_w2v.wv.vocab.items()}
    indexed_vocab_new = sorted([k for k in indexed_vocab.items()], key = lambda x:x[0])
    indexed_vocab_new = [x[1] for x in indexed_vocab_new]

    def plot_embedding_plt(X, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure(figsize=(50, 50), dpi=300)
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(indexed_vocab[i]),
                     fontdict={'size': 20})

        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

    def plot_embedding_bokeh(X, width = 1000, height = 1000):
        from bokeh.models import HoverTool
        hover = HoverTool(tooltips = [
                ('Word', '@target'),
                ('index', '$index')
        ])
        
        p = bokeh.plotting.figure(plot_width=width, plot_height=height, tools = [hover, bokeh.models.WheelZoomTool(), bokeh.models.PanTool()])
        df = pd.DataFrame(X, columns=['x', 'y'])
        df['target'] = pd.Series(indexed_vocab_new, index=df.index)

        source = bokeh.plotting.ColumnDataSource(df)
        p.circle(x = df.x, y = df.y, source = source, size=20, color="navy", alpha=0.5)
        return p
        show(plot_embedding_bokeh(tsne_vectors, width = 900, height = 700))
    plot_embedding_plt(tsne_vectors)
    plt.savefig('yes.png')


def get_all_datasetnames(dataset_folder=DATASET_FOLDER):
    return set([filename.split('/')[1].split('-')[0] for filename in glob(r'{}/*.txt'.format(DATASET_FOLDER))])


def _get_data_container():
    data_container = base.Bunch()
    data_container.target_names = []
    data_container.data = []
    return data_container


def get_dataset(dataset, subset, typ='stemmed', category_filter=None, dataset_folder=DATASET_FOLDER):
    if dataset not in CACHE:
        CACHE[dataset] = {}
    if typ not in CACHE[dataset] or subset not in CACHE[dataset][typ]:
        with codecs.open('{}/{}-{}-{}.txt'.format(dataset_folder, dataset, subset, typ), encoding='utf-8') as f:
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


def plot_confusion_matrix(cm,
                          classes = None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          round_confusion=2,
                          x_rotation=90):
    """
    Plots the confusion matrix.
    Taken from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
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
