import yaml
import os
import transformers
from transformers.pipelines import pipeline_helper
import experiments
from experiments import task_helper
from utils import dataset_helper, graph_metrics
import sklearn
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.feature_extraction.text
from glob import glob
from utils import constants
import collections
from nltk.stem import LancasterStemmer


NEEDED_FIELDS = ['params_per_type']

# These are field values for experiment yaml files that get replaced with their corresponding class.
# For example, "SVC" will be replaced with an instance of sklearn.svm.SCV
PLACEHOLDER_LIST = dict(
    LancasterStemmer=LancasterStemmer,
    nxgraph_degrees_metric=graph_metrics.nxgraph_degrees_metric,
    nxgraph_degrees_metric_max=graph_metrics.nxgraph_degrees_metric_max,
    nxgraph_pagerank_metric=graph_metrics.nxgraph_pagerank_metric,
    adj_degrees_metric=graph_metrics.adj_degrees_metric,
    adj_degrees_metric_max=graph_metrics.adj_degrees_metric_max,
    MaxAbsScaler=sklearn.preprocessing.MaxAbsScaler,
    DummyClassifier=sklearn.dummy.DummyClassifier,
    SVC=sklearn.svm.SVC,
    PCA=sklearn.decomposition.PCA,
    TruncatedSVD=sklearn.decomposition.TruncatedSVD,
    iteration_weight_function=transformers.fast_wl_graph_kernel_transformer.iteration_weight_function,
)

# Add the names of the own transformers
PLACEHOLDER_LIST = dict(PLACEHOLDER_LIST, **transformers.TRANSFORMERS)


def get_experiment_config(file: str = constants.EXPERIMENT_CONFIG_ALL) -> dict:
    assert os.path.exists(file)

    with open(file) as f:
        experiment_options = yaml.load(f.read())

    for field in NEEDED_FIELDS:
        assert field in experiment_options, 'Missing field: {}'.format(field)

    task_type_params = {}
    for task_type, params in experiment_options['params_per_type'].items():
        flattened_params = pipeline_helper.flatten_nested_params(params)
        task_type_params[task_type] = replace_placeholders(flattened_params)

    experiment_name = experiment_options.get('experiment_name', None)
    filename = file.split('/')[-1]

    # Use filename as experiment_name if not explicit experiment_name is given
    if not experiment_name:
        experiment_name = filename.rsplit('.', 1)[0]

    return dict(
        experiment_name=experiment_name,
        params_per_type=task_type_params,
        limit_dataset=experiment_options.get('limit_dataset', None),
        limit_graph_type=experiment_options.get('limit_graph_type', None),
        filename=filename
    )


def get_all_task_type_params():
    task_type_params = {}
    for task in experiments.get_all_tasks():
        if task.type in task_type_params: continue
        X, Y, estimator, params = task.fn()
        task_type_params[task.type] = params
        del X, Y, estimator, params
    return task_type_params


def prepare_param_grid(task, param_grid, experiment_config):
    experiment_params = dict()
    if experiment_config:
        assert task.type in experiment_config['params_per_type']
        experiment_params = pipeline_helper.flatten_nested_params(experiment_config['params_per_type'][task.type])

    param_grid = pipeline_helper.flatten_nested_params(param_grid)

    # Merge param_grid with classifiers
    param_grid = task_helper.add_classifier_to_params(param_grid)

    # Overwrite default param_grid with the parameters specified in experiment_config
    param_grid = dict(param_grid, **experiment_params)

    # Remove "voided" params
    param_grid = {k: v for k, v in param_grid.items() if v is not None}

    # Instantiate classes, eg. SVM or RelabelTransformer
    param_grid = {k: [x() if isinstance(x, type) else x for x in v] for k, v in param_grid.items()}
    return param_grid


def get_all_task_typ_params_flat(task_type_params: dict = None, remove_complex_types=True):
    if not task_type_params:
        task_type_params = get_all_task_type_params()

    clean_params_config = {}
    for k, v in task_type_params.items():
        out = pipeline_helper.flatten_nested_params(v)
        keys = out.keys()
        non_dump_keys = []
        for key1 in keys:
            for key2 in keys:
                if key1 == key2: continue
                if key1.startswith(key2):
                    non_dump_keys.append(key2)
        if remove_complex_types:
            out = pipeline_helper.remove_complex_types(out)
        non_dump_keys = set(non_dump_keys)

        clean_params = {k: v for k, v in out.items() if k not in non_dump_keys}
        for key in non_dump_keys:
            clean_params[key + '__VAL_'] = out[key]

        clean_unflattened_params = pipeline_helper.unflatten_params(clean_params)
        clean_params_config[k] = clean_unflattened_params

    return clean_params_config


def save_experiment_params_as_experiment_config(file: str = constants.EXPERIMENT_CONFIG_ALL, only_with_concept_maps: bool=True):
    folder = file.rsplit('/', 1)[0]
    os.makedirs(folder, exist_ok=True)

    task_type_params = get_all_task_typ_params_flat()

    if only_with_concept_maps:
        datasets = dataset_helper.get_dataset_names_with_concept_map()
    else:
        datasets = dataset_helper.get_all_available_dataset_names()

    with open(file, 'w') as f:
        yaml.dump(dict(
            limit_graph_type=constants.GRAPH_TYPES,
            limit_dataset=datasets,
            params_per_type=task_type_params
        ), f, default_flow_style=False)


def replace_placeholders(param_grid, placeholder_list=PLACEHOLDER_LIST):
    out_param_grid = {}
    for k, val in param_grid.items():
        if val is None:
            out_param_grid[k] = val
            continue
        assert isinstance(val, list)
        outs = []
        for el in val:
            if el in placeholder_list:
                placeholder_val = placeholder_list[el]
                outs.append(placeholder_val)
                continue
            outs.append(el)
        out_param_grid[k] = outs
    return out_param_grid


def get_all_param_grid_config_files(folder=constants.EXPERIMENT_CONFIG_FOLDER, filter_out_disabled=True):
    out = {}
    for file in glob('{}/**/*.yaml'.format(folder), recursive=True):
        if filter_out_disabled and '.disabled.' in file or 'all_experiments' in file:
            continue
        out[file] = get_experiment_config(file)
    return out

def get_all_experiment_names():
    experiments_names = [x.split('/')[-1].rsplit('.', 1)[0] for x in get_all_param_grid_config_files().keys() if not x.endswith('all.yaml') and not x.endswith('all_experiments.yaml')]
    return experiments_names


def get_experiment_config_for(experiment_name:str, folder=constants.EXPERIMENT_CONFIG_FOLDER):
    file = '{}/{}.yaml'.format(folder, experiment_name)
    assert os.path.exists(file)
    return get_experiment_config(file)


def save_all_experiment_params():
    all_tasks = experiments.get_all_tasks()

    tasks = {}
    for task in all_tasks:
        if task.name in tasks: continue
        tasks[task.type] = task

    all_experiments = get_all_param_grid_config_files()

    experiments_ = collections.defaultdict(dict)
    for name, experiment_config in all_experiments.items():
        if '/all' in name: continue
        for task_name, task in tasks.items():
            if task.type not in experiment_config['params_per_type']: continue
            _, _, _, params = task.fn()
            merged_param_grid = prepare_param_grid(task, params, experiment_config)
            experiments_[name][task.type] = merged_param_grid

    out = {}
    for experiment_config, param_vals in experiments_.items():
        a = get_all_task_typ_params_flat(param_vals)
        out[experiment_config.split('/', 2)[2]] = a

    with open('configs/experiments/all_experiments.yaml', 'w') as f:
        yaml.dump(out, f, default_flow_style=False)