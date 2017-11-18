from transformers.graph_to_text_transformer import GraphToTextTransformer
from transformers.pipelines import text_pipeline, graph_pipeline
from utils import dataset_helper, graph_helper
from utils.filename_utils import get_filename_only
from . import task_helper
from .task_helper import ExperimentTask, ClassificationData
import sklearn
from sklearn import dummy, model_selection, preprocessing, pipeline


def get_tasks() -> list:
    graph_cache_files = dataset_helper.get_all_cached_graph_datasets()
    gram_cache_files = dataset_helper.get_all_gram_datasets()
    datasets = dataset_helper.get_all_available_dataset_names()

    graph_task_fns = [
        get_task_graph_content_only,
        get_task_graph_structure_only,
        get_task_combined,
        get_task_graphs
    ]

    tasks = []

    tasks += task_helper.get_tasks(graph_task_fns, graph_cache_files)
    tasks += task_helper.get_tasks([get_task_dummy], datasets)
    tasks += task_helper.get_tasks([get_task_text], datasets)
    tasks += task_helper.get_tasks([get_gram_task], gram_cache_files)

    return tasks


def get_task_dummy(dataset_name: str) -> ExperimentTask:
    dummy_classifier = [sklearn.dummy.DummyClassifier()]
    dummy_classifier_strategy = ['most_frequent', 'stratified', 'uniform']
    vectorizer = sklearn.feature_extraction.text.CountVectorizer()

    tasks = []
    for strategy in dummy_classifier_strategy:
        def process():
            X, Y = dataset_helper.get_dataset(dataset_name)
            estimator = sklearn.pipeline.Pipeline([('vectorizer', vectorizer), ('classifier', None)])
            params = dict(classifier=dummy_classifier, classifier__strategy=[strategy], classifier__C = None, classifier__max_iter = None, classifier__tol = None)
            return ClassificationData(X, Y, estimator, params)

        tasks.append(ExperimentTask('dummy_{}'.format(strategy), dataset_name, process))
    return tasks


def get_task_graph_content_only(graph_cache_file) -> ExperimentTask:
    def process():
        X, Y = dataset_helper.get_dataset_cached(graph_cache_file)
        estimator, params = text_pipeline.get_params(reduced=False)
        estimator.steps.insert(0, ('graph_to_text', GraphToTextTransformer()))
        params = dict(params, **dict(graph_to_text__use_edges=[True, False]))
        return ClassificationData(X, Y, estimator, params)

    return ExperimentTask('graph_content_only', get_filename_only(graph_cache_file), process)


def get_task_graph_structure_only(graph_cache_file) -> ExperimentTask:
    def process() -> tuple:
        X, Y = dataset_helper.get_dataset_cached(graph_cache_file)
        X = graph_helper.get_graphs_only(X)
        graph_helper.convert_graphs_to_adjs_tuples(X)
        X = [(x, [0] * len(y)) for x, y in X]
        estimator, params = task_helper.get_graph_estimator_and_params(X, Y)
        return ClassificationData(X, Y, estimator, params)

    return ExperimentTask('graph_structure_only', get_filename_only(graph_cache_file), process)


def get_task_text(dataset_name: str) -> ExperimentTask:
    def process() -> tuple:
        X, Y = dataset_helper.get_dataset(dataset_name)
        estimator, params = text_pipeline.get_params(reduced=False)
        return ClassificationData(X, Y, estimator, params)

    return ExperimentTask('text', dataset_name, process)


def get_task_combined(graph_cache_file: str) -> ExperimentTask:
    def process() -> tuple:
        X, Y = graph_helper.get_combined_text_graph_dataset(graph_cache_file)

        graphs = [g for (g, _, _) in X]
        graph_helper.convert_graphs_to_adjs_tuples(graphs)
        num_vertices = task_helper.get_num_vertices(graphs)

        X = [(graph, text) for (graph, text, _) in X]

        estimator, params = graph_pipeline.get_combined_params(text_reduced=True)
        graph_pipeline.add_num_vertices_to_fast_wl_params(params, num_vertices=num_vertices)

        return ClassificationData(X, Y, estimator, params)

    return ExperimentTask('graph_combined', get_filename_only(graph_cache_file), process)


def get_task_graphs(graph_cache_file: str) -> ExperimentTask:
    def process() -> tuple:
        X, Y = dataset_helper.get_dataset_cached(graph_cache_file)
        graph_helper.convert_graphs_to_adjs_tuples(X, copy=False)
        estimator, params = task_helper.get_graph_estimator_and_params(X, Y)
        return ClassificationData(X, Y, estimator, params)

    return ExperimentTask('graph', get_filename_only(graph_cache_file), process)


def get_gram_task(gram_cache_file) -> ExperimentTask:
    def process() -> tuple:
        K, Y = dataset_helper.get_dataset_cached(gram_cache_file, check_validity=False)
        estimator = sklearn.pipeline.Pipeline([('classifier', None)])
        params = dict(classifier=[sklearn.svm.SVC(kernel='precomputed', class_weight='balanced')])
        return ClassificationData(K, Y, estimator, params)

    return ExperimentTask('graph_gram', get_filename_only(gram_cache_file), process)
