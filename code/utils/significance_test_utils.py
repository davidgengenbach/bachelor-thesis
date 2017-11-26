import sklearn
from sklearn import preprocessing, metrics as m
import numpy as np
import functools
import matplotlib.pyplot as plt
import pandas as pd
import collections
import typing

MetricFunction = typing.Callable[[typing.Iterable, typing.Iterable], float]

Result = collections.namedtuple('Result', ['y_true', 'y_preds'])

accuracy: MetricFunction = sklearn.metrics.accuracy_score
recall: MetricFunction = functools.partial(sklearn.metrics.recall_score, average='macro')
f1: MetricFunction = functools.partial(sklearn.metrics.f1_score, average='macro')

metrics: typing.Iterable[typing.Tuple[str, MetricFunction]] = [('accuracy', accuracy), ('recall_macro', recall), ('f1_macro', f1)]


def get_transformed_results(result: Result):
    y_true = result.y_true
    y_pred_a, y_pred_b = result.y_preds
    trans_enc = sklearn.preprocessing.LabelEncoder()
    y_true = trans_enc.fit_transform(y_true)
    y_pred_a, y_pred_b = trans_enc.transform(y_pred_a), trans_enc.transform(y_pred_b)
    return np.array(y_true), np.array(y_pred_a), np.array(y_pred_b)


def randomization_test(y_true: typing.Iterable, y_pred_a: typing.Iterable, y_pred_b: typing.Iterable, metric: MetricFunction=sklearn.metrics.f1_score, num_trails:int=1000):
    y_true, y_pred_a, y_pred_b = np.array(y_true), np.array(y_pred_a), np.array(y_pred_b)
    metrics_ = np.empty((num_trails, 2), dtype=np.float64)

    def get_shuffled_array(indices_, to_compare=0):
        other_to_compare = (to_compare + 1) % 2
        y_shuffled = np.empty(len(y_true), dtype=y_true.dtype)
        y_shuffled[indices_ == to_compare] = y_pred_a[indices_ == to_compare]
        y_shuffled[indices_ == other_to_compare] = y_pred_b[indices_ == other_to_compare]
        return y_shuffled

    for i in range(num_trails):
        # Generate a array with random 0s and 1s
        # 0 means that the value at that index gets taken from y_pred_a, 1 means y_pred_b
        indices = np.random.randint(0, 2, size=len(y_true))
        # Pick an element either from y_pred_a or y_pred_b (= interchange them), depending on the value of the indices array
        y_shuffled_a, y_shuffled_b = get_shuffled_array(indices, to_compare=0), get_shuffled_array(indices, to_compare=1)
        # Calculate metric for both models (with interchanged elements)
        metric_a, metric_b = metric(y_true, y_shuffled_a), metric(y_true, y_shuffled_b)
        metrics_[i, :] = [metric_a, metric_b]
    return metrics_


def get_confidence(diff_global: float, diffs: typing.Iterable, num_trails: int):
    return (np.sum(np.fabs(diffs) >= np.fabs(diff_global)) + 1) / (num_trails + 1)


def plot_randomzation_test_distribution(result: Result, metric: MetricFunction=f1, metric_name: str='NOT_SET', num_trails: int=1000):
    y_true, y_pred_a, y_pred_b = get_transformed_results(result)
    metric_a, metric_b = metric(y_true, y_pred_a), metric(y_true, y_pred_b)
    # Global diff
    diff = metric_a - metric_b
    # Randomization test
    metrics_ = randomization_test(y_true, y_pred_a, y_pred_b, metric=metric, num_trails=num_trails)

    # Calculate diffs for the randomized results
    diffs = metrics_[:, 0] - metrics_[:, 1]

    # Get confidence (= the probability that the observed difference between the metrics on model A and B is a product of chance)
    p = get_confidence(diff, diffs, num_trails=num_trails)

    # Plot data
    fig, ax = plt.subplots()
    df = pd.DataFrame({'metric': diffs})
    df.metric.plot(kind='hist', bins=100, ax=ax, title='Metric: {}, p={:.4f}, #trails={}, diff={:.4f}'.format(metric_name, p, num_trails, diff))

    for x, color in [(diff, 'red'), (-diff, 'red'), (diffs.mean(), 'green')]:
        ax.axvline(x, color=color)

    plt.show()
    plt.close(fig)
    return fig, ax

def plot_randomization_test_distribution_(diffs, global_diff, num_trails='NOT_SET', p=None, metric_name: str='NOT_SET', ax=None):
    # Plot data
    if not ax:
        _, ax = plt.subplots()

    fig = ax.get_figure()

    df = pd.DataFrame({'metric': diffs})
    df.metric.plot(kind='hist', bins=100, ax=ax, title='Metric: {}, p={:.4f}, #trails={}, diff={:.4f}'.format(metric_name, p, num_trails, global_diff))

    for x, color in [(global_diff, 'red'), (-global_diff, 'red'), (diffs.mean(), 'green')]:
        ax.axvline(x, color=color)

    fig.tight_layout()
    return fig, ax