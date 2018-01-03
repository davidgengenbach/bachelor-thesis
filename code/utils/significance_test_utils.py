import sklearn
from sklearn import preprocessing, metrics as m
import numpy as np
import functools
import matplotlib.pyplot as plt
import pandas as pd
import collections
import typing
from joblib import Parallel, delayed
import os
import pickle

MetricFunction = typing.Callable[[typing.Iterable, typing.Iterable], float]

Result = collections.namedtuple('Result', ['y_true', 'y_preds'])

accuracy: MetricFunction = sklearn.metrics.accuracy_score
recall: MetricFunction = functools.partial(sklearn.metrics.recall_score, average='macro')
f1: MetricFunction = functools.partial(sklearn.metrics.f1_score, average='macro')

metrics: typing.Iterable[typing.Tuple[str, MetricFunction]] = [('accuracy', accuracy), ('recall_macro', recall), ('f1_macro', f1)]


def get_confidences(df, model_selection_attr, model_selection_vals, performance_attr='prediction_score_f1_macro', log_progress=None, **test_params):
    data = collections.defaultdict(list)
    for dataset, df_ in log_progress(df.groupby('dataset')) if log_progress else df.groupby('dataset'):
        best = df_.loc[df_.groupby(model_selection_attr)[performance_attr].idxmax()]

        if len(best) != 2:
            print('\tToo many/too few models. Got {}, expected 2. Skipping. Dataset: {}'.format(len(best), dataset))
            continue

        prediction_filenames = [best.loc[best[model_selection_attr] == name].iloc[0].prediction_file for name in model_selection_vals]

        diffs, score_a, score_b, global_difference, confidence = calculate_significance(prediction_filenames[0], prediction_filenames[1], **test_params)

        data['dataset'].append(dataset)
        data['filenames'].append(prediction_filenames)

        data['diffs'].append(diffs)
        data['scores'].append([score_a, score_b])
        data['global_difference'].append(global_difference)

        data['confidence'].append(confidence)
    return pd.DataFrame(data).set_index('dataset')


def calculate_significance(prediction_file_a, prediction_file_b, n_jobs=5, num_trails=5000, one_tail=False):
    models = []
    for idx, file in enumerate([prediction_file_a, prediction_file_b]):
        if not os.path.exists(file):
            raise FileNotFoundError('Not found: {}'.format(file))

        with open(file, 'rb') as f:
            predictions = pickle.load(f)

        res = predictions['results']
        Y_real, Y_pred, Y_test = [res[x] for x in ('Y_real', 'Y_pred', 'X_test')]
        models.append(dict(
            Y_real=Y_real,
            Y_pred=Y_pred,
            Y_test=Y_test
        ))

    model_a, model_b = models

    if not np.array_equal(model_a['Y_real'], model_b['Y_real']):
        raise Exception('Invalid models to compare: the Y_real labels must be the same for both labels!')

    Y_real = model_a['Y_real']

    test_result = randomization_test(y_true=Y_real, y_pred_a=models[0]['Y_pred'], y_pred_b=models[1]['Y_pred'], num_trails=num_trails, n_jobs=n_jobs, one_tail=one_tail)
    diffs, score_a, score_b, global_difference, confidence = test_result
    return test_result


def get_transformed_results(result: Result):
    y_true = result.y_true
    y_pred_a, y_pred_b = result.y_preds
    trans_enc = sklearn.preprocessing.LabelEncoder()
    y_true = trans_enc.fit_transform(y_true)
    y_pred_a, y_pred_b = trans_enc.transform(y_pred_a), trans_enc.transform(y_pred_b)
    return np.array(y_true), np.array(y_pred_a), np.array(y_pred_b)


def _get_shuffled_array(indices_, y_true, y_pred_a, y_pred_b, to_compare=0):
    other_to_compare = (to_compare + 1) % 2
    y_shuffled = np.empty(len(y_true), dtype=y_true.dtype)
    y_shuffled[indices_ == to_compare] = y_pred_a[indices_ == to_compare]
    y_shuffled[indices_ == other_to_compare] = y_pred_b[indices_ == other_to_compare]
    return y_shuffled


def _get_permutated_result(y_true, y_pred_a, y_pred_b, metric):
    # Generate a array with random 0s and 1s
    # 0 means that the value at that index gets taken from y_pred_a, 1 means y_pred_b
    indices = np.random.randint(0, 2, size=len(y_true))
    # Pick an element either from y_pred_a or y_pred_b (= interchange them), depending on the value of the indices array
    y_shuffled_a, y_shuffled_b = _get_shuffled_array(indices, y_true, y_pred_a, y_pred_b, to_compare=0), _get_shuffled_array(indices, y_true, y_pred_a, y_pred_b, to_compare=1)
    # Calculate metric for both models (with interchanged elements)
    metric_a, metric_b = metric(y_true, y_shuffled_a), metric(y_true, y_shuffled_b)
    return metric_a, metric_b


def randomization_test(y_true: typing.Iterable, y_pred_a: typing.Iterable, y_pred_b: typing.Iterable, metric: MetricFunction = f1, num_trails: int = 1000, n_jobs: int = 1, one_tail: bool = True):
    y_true, y_pred_a, y_pred_b = np.array(y_true), np.array(y_pred_a), np.array(y_pred_b)

    results = Parallel(n_jobs=n_jobs)(delayed(_get_permutated_result)(y_true, y_pred_a, y_pred_b, metric) for i in range(num_trails))
    metrics_ = np.array(results, dtype=np.float64)
    diffs = metrics_[:, 0] - metrics_[:, 1]
    score_a = metric(y_true, y_pred_a)
    score_b = metric(y_true, y_pred_b)
    global_difference = score_a - score_b
    confidence = get_confidence(diff_global=global_difference, diffs=diffs, num_trails=num_trails, one_tail=one_tail)
    return diffs, score_a, score_b, global_difference, confidence


def get_confidence(diff_global: float, diffs: typing.Iterable, num_trails: int, one_tail: bool = True):
    if one_tail:
        if diff_global < 0:
            diff_global = -diff_global
            diffs = -diffs
    else:
        diffs = np.fabs(diffs)
        diff_global = np.fabs(diff_global)

    return (np.sum(diffs >= diff_global) + 1) / (num_trails + 1)


def plot_randomzation_test_distribution(result: Result, metric: MetricFunction = f1, metric_name: str = 'NOT_SET', num_trails: int = 1000):
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

    for x, color in [(diff, 'red'), (-diff, 'blue'), (diffs.mean(), 'green')]:
        ax.axvline(x, color=color)

    plt.show()
    plt.close(fig)
    return fig, ax


def plot_randomization_test_distribution_(diffs, global_diff, num_trails='NOT_SET', p=None, metric_name: str = 'NOT_SET', ax=None):
    # Plot data
    if not ax:
        _, ax = plt.subplots()

    fig = ax.get_figure()

    df = pd.DataFrame({'metric': diffs})
    df.metric.plot(kind='hist', bins=100, ax=ax, title='Metric: {}, p={:.4f}, #trails={}, diff={:.4f}'.format(metric_name, p, num_trails, global_diff))

    for x, color in [(global_diff, 'red'), (-global_diff, 'blue'), (diffs.mean(), 'green')]:
        ax.axvline(x, color=color)

    fig.tight_layout()
    return fig, ax
