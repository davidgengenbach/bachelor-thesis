import sklearn
from glob import glob
import collections
import networkx as nx
import argparse
import pickle
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import os

from transformers import text_pipeline
from transformers import fast_wl_pipeline
from transformers.tuple_selector import TupleSelector
from utils.logger import LOGGER
from utils.remove_coefs_from_results import remove_coefs_from_results
from utils import dataset_helper, filename_utils, time_utils, git_utils, graph_helper

Task = collections.namedtuple('Task', ['type', 'name', 'process_fn', 'process_fn_args'])


def get_all_classification_tasks(args, clfs=None):
    tasks = []
    tasks += get_text_classification_tasks(args, clfs)
    tasks += get_graph_classification_tasks(args, clfs)
    tasks += get_gram_classification_tasks(args, clfs)
    return tasks


def get_gram_classification_tasks(args: argparse.Namespace, clfs):
    tasks = []

    def process(args: argparse.Namespace, task: Task, gram_cache_file: str):
        K, Y = dataset_helper.get_dataset_cached(gram_cache_file, check_validity=False)
        estimator = sklearn.pipeline.Pipeline([('classifier', sklearn.svm.SVC(kernel='precomputed', class_weight='balanced', max_iter=args.max_iter, tol=args.tol))])
        cross_validate(args, task, K, Y, estimator, {}, is_precomputed=True)

    # Gram classification
    for gram_cache_file in glob('data/CACHE/*gram*.npy'):
        filename = filename_utils.get_filename_only(gram_cache_file)
        tasks.append(Task(type='graph_gram', name=filename, process_fn=process, process_fn_args=[gram_cache_file]))

    return tasks


def get_graph_classification_tasks(args: argparse.Namespace, clfs):
    tasks = []

    graph_fast_wl_classification_pipeline = sklearn.pipeline.Pipeline([
        ('feature_extraction', fast_wl_pipeline.get_pipeline()),
        ('classifier', None)
    ])

    graph_fast_wl_grid_params = {
        'fast_wl__h': args.wl_iterations,
        'fast_wl__phi_dim': [None],
        'fast_wl__round_to_decimals': args.wl_round_to_decimal,
        'phi_picker__return_iteration': args.wl_phi_picker_iterations
    }

    def process_(args: argparse.Namespace, task: Task, X, Y):
        fast_wl_pipeline.convert_graphs_to_tuples(X)

        empty_graphs = [1 for _, labels in X if len(labels) == 0]
        num_vertices = sum([len(labels) for _, labels in X]) + len(empty_graphs)

        features_params = dict(
            {'feature_extraction__' + k: val for k, val in graph_fast_wl_grid_params.items()},
            **dict(feature_extraction__fast_wl__phi_dim=[num_vertices])
        )

        grid_params_scratch = dict(dict(classifier=clfs), **features_params)

        cross_validate(args, task, X, Y, graph_fast_wl_classification_pipeline, grid_params_scratch)

    def process_plain(args: argparse.Namespace, task: Task, graph_cache_file: str):
        X, Y = dataset_helper.get_dataset_cached(graph_cache_file)
        process_(args, task, X, Y)

    def process_same_label(args: argparse.Namespace, task: Task, graph_cache_file: str):
        X, Y = dataset_helper.get_dataset_cached(graph_cache_file)
        fast_wl_pipeline.convert_graphs_to_tuples(X)
        X_same_label = [(x, [0] * len(y)) for x, y in X]
        process_(args, task, X_same_label, Y)

    def process_tfidf_graphs(args: argparse.Namespace, task: Task, graph_cache_file: str):
        X, Y = dataset_helper.get_dataset_cached(graph_cache_file)
        graph_helper.remove_graph_labels(X)
        X = [graph_helper.graph_to_text(g) for g in X]

        cross_validate(args, task, X, Y, text_pipeline.get_pipeline(), dict(text_pipeline.get_param_grid(), **dict(classifier=clfs)))


    def process_combined(args: argparse.Namespace, task: Task, graph_cache_file: str):
        X_combined, Y_combined = graph_helper.get_filtered_text_graph_dataset(graph_cache_file)

        graphs = [g for (g, _, _) in X_combined]
        empty_graphs = len([1 for g in graphs if nx.number_of_nodes(g) == 0 or nx.number_of_edges(g) == 0])
        num_vertices = sum([nx.number_of_nodes(g) for g in graphs]) + empty_graphs
        fast_wl_pipeline.convert_graphs_to_tuples(graphs)

        X_combined = [(graph, text) for (graph, text, _) in X_combined]

        grid_params_combined = dict({
            'classifier': clfs
        }, **dict({'features__fast_wl_pipeline__feature_extraction__' + k: val for k, val in
                   graph_fast_wl_grid_params.items()}, **dict(
            features__fast_wl_pipeline__feature_extraction__fast_wl__phi_dim=[num_vertices]
        )))

        combined_features = sklearn.pipeline.FeatureUnion([
            ('tfidf', sklearn.pipeline.Pipeline([
                ('selector', TupleSelector(tuple_index=1)),
                ('tfidf', text_pipeline.get_pipeline()),
            ])),
            ('fast_wl_pipeline', sklearn.pipeline.Pipeline([
                ('selector', TupleSelector(tuple_index=0, v_stack=False)),
                ('feature_extraction', fast_wl_pipeline.get_pipeline())
            ]))
        ])

        pipeline = sklearn.pipeline.Pipeline([
            ('features', combined_features),
            ('classifier', None)
        ])

        return cross_validate(args, task, X_combined, Y_combined, pipeline, grid_params_combined)

    for graph_cache_file in dataset_helper.get_all_cached_graph_datasets():
        filename = filename_utils.get_filename_only(graph_cache_file)

        tasks.append(Task(type='graph_tfidf', name=filename, process_fn=process_tfidf_graphs, process_fn_args=[graph_cache_file]))
        tasks.append(Task(type='graph_fast_wl', name=filename, process_fn=process_plain, process_fn_args=[graph_cache_file]))
        tasks.append(Task(type='graph_fast_wl_same_label', name=filename, process_fn=process_same_label, process_fn_args=[graph_cache_file]))
        tasks.append(Task(type='graph_combined-fast_wl', name=filename, process_fn=process_combined, process_fn_args=[graph_cache_file]))

    return tasks


def get_text_classification_tasks(args: argparse.Namespace, clfs):
    tasks = []

    def process(args: argparse.Namespace, task: Task, dataset_name: str):
        X, Y = dataset_helper.get_dataset(dataset_name)
        cross_validate(args, task, X, Y, text_pipeline.get_pipeline(), dict(text_pipeline.get_param_grid(), **dict(classifier=clfs)))

    # Text classification
    for dataset_name in dataset_helper.get_all_available_dataset_names():
        tasks.append(Task(type='text', name=dataset_name, process_fn=process, process_fn_args=[dataset_name]))

    return tasks


def get_precomputed_subset(K, indices1, indices2=None):
    if not indices2:
        indices2 = indices1
    indices = np.meshgrid(indices1, indices2, indexing='ij')
    return np.array(K)[indices]


def train_test_split(X, Y, test_size: float=0.15, random_state: int = 42, is_precomputed: bool=False):
    def train_test_split(*Xs, Y = None):
        return sklearn.model_selection.train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=random_state)

    if is_precomputed:
        # Cut the precomputed gram matrix into a train/test split...
        num_elements = X.shape[0]
        indices = list(range(num_elements))
        # ... by first splitting the dataset into train/test indices
        X_train_i, X_test_i = train_test_split(indices, Y=Y)
        # ... then cut the corresponding elements from the gram matrix
        X_train, Y_train = get_precomputed_subset(X, X_train_i), np.array(Y)[X_train_i]
        X_test, Y_test = get_precomputed_subset(X, X_test_i, X_train_i), np.array(Y)[X_test_i]
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, Y=Y)

    return X_train, X_test, Y_train, Y_test


def cross_validate(args: argparse.Namespace, task: Task, X, Y, estimator, param_grid: dict, skip_predictions=False, is_precomputed=False):
    cv = sklearn.model_selection.StratifiedKFold(
        n_splits=args.n_splits,
        random_state=args.random_state,
        shuffle=True
    )

    result_filename_tmpl = filename_utils.get_result_filename_for_task(task)

    result_file = '{}/{}'.format(args.results_folder, result_filename_tmpl)
    predictions_file = '{}/{}'.format(args.predictions_folder, result_filename_tmpl)

    if not args.force and os.path.exists(result_file):
        return

    X_train, Y_train, X_test, Y_test = X, Y, [], []


    if not skip_predictions and args.create_predictions:
        # Hold out validation set for predictions
        try:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.prediction_test_size, random_state=args.random_state, is_precomputed=is_precomputed)
        except Exception as e:
            LOGGER.warning('Could not split dataset for predictions')
            LOGGER.exception(e)

    gscv = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=args.scoring, n_jobs=args.n_jobs, verbose=args.verbose, refit=args.refit)

    gscv_result = gscv.fit(X_train, Y_train)

    if not skip_predictions and args.create_predictions:
        if not len(X_test):
            LOGGER.warning('Validation set for prediction has no length: len(X_test)={}'.format(len(X_test)))
        else:
            try:
                # Retrain the best classifier and get prediction on validation set
                best_classifier = sklearn.base.clone(gscv_result.best_estimator_)
                best_classifier.fit(X_train, Y_train)
                Y_test_pred = best_classifier.predict(X_test)
                dump_pickle_file(args, predictions_file, {
                    'results': {
                        'Y_real': Y_test,
                        'Y_pred': Y_test_pred,
                        'X_test': X_test
                    }
                })

            except Exception as e:
                LOGGER.warning('Error while trying to retrain best classifier')
                LOGGER.exception(e)

    if not args.keep_coefs:
        remove_coefs_from_results(gscv_result.cv_results_)

    dump_pickle_file(args, result_file, dict(
        results=gscv_result.cv_results_
    ))


def dump_pickle_file(args, filename: str, data: dict, add_meta: bool = True):
    meta = dict(meta_data=get_metadata(args)) if add_meta else dict()

    with open(filename, 'wb') as f:
        pickle.dump(dict(meta, **data), f)


def get_metadata(args, other=None) -> dict:
    return dict(
        git_commit=str(git_utils.get_current_commit()),
        timestamp=time_utils.get_timestamp(),
        timestamp_readable=time_utils.get_time_formatted(),
        args=vars(args),
        other=other
    )
