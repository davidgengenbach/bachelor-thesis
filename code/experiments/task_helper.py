from transformers.pipelines import graph_pipeline
from transformers.pipelines import classifiers
import networkx as nx
import collections
import sklearn
from utils import graph_helper
import typing

ExperimentTask = collections.namedtuple('ExperimentTask', ['type', 'name', 'fn'])
ClassificationData = collections.namedtuple('ClassificationData', ['X', 'Y', 'estimator', 'params'])


def get_tasks(tasks_fns: typing.Union[ExperimentTask, typing.Iterable[ExperimentTask]], args: typing.Iterable):
    tasks = []
    for arg in args:
        for task_fn in tasks_fns:
            res = task_fn(arg)

            if isinstance(res, ExperimentTask):
                res = [res]

            tasks += res
    return tasks


def get_num_vertices(X: typing.List) -> int:
    '''
    Returns the total number of vertices in the graphs
    TODO: This has to go into utils.graph_helper

    Args:
        X: list of tuples (adj, labels)

    Returns:
        the total number of vertices in the graphs
    '''
    empty_graphs = [1 for _, labels in X if len(labels) == 0]
    num_vertices = sum([len(labels) for _, labels in X]) + len(empty_graphs)
    return num_vertices


def get_graph_estimator_and_params(X, Y=None, reduced: bool=False, with_node_weights: bool=False, add_phi_dim=False):
    assert len(X)
    X_ = X

    estimator, params = graph_pipeline.get_params(reduced=reduced, with_node_weights=with_node_weights)

    if add_phi_dim:
        if isinstance(X[0], nx.Graph):
            X_ = graph_helper.convert_graphs_to_adjs_tuples(X, copy=True)
        assert isinstance(X_[0], tuple) and isinstance(X_[0][1], list)
        num_vertices = get_num_vertices(X_) * 5
        graph_pipeline.add_num_vertices_to_fast_wl_params(params, num_vertices)

    return estimator, params


def test_params(params: dict):
    # This raises when passing invalid params
    params_ = sklearn.model_selection.ParameterGrid(params)

    for k, v in params_.items():
        assert isinstance(v, list)
        v = [v if isinstance(v, (int, float, str, bool, tuple)) else type(v).__name__ for v in v]


def add_classifier_to_params(params: dict) -> dict:
    classifier_params = classifiers.get_classifier_params()
    param_grid = dict(classifier_params, **params)
    return param_grid
