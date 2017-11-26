import yaml
import os
from transformers.simple_preprocessing_transformer import SimplePreProcessingTransformer
from transformers.pipelines import pipeline_helper
import experiments
from experiments import task_helper
from utils import dataset_helper, graph_metrics
import sklearn
import sklearn.preprocessing
import sklearn.feature_extraction.text
from glob import glob

NEEDED_FIELDS = ['params_per_type']

EXPERIMENT_CONFIG_FOLDER = 'configs/param_grid_configs'
EXPERIMENT_CONFIG_ALL = EXPERIMENT_CONFIG_FOLDER + '/all.yaml'


PLACEHOLDER_LIST = dict(
    nxgraph_degrees_metric=graph_metrics.nxgraph_degrees_metric,
    nxgraph_pagerank_metric=graph_metrics.nxgraph_pagerank_metric,
    SimplePreProcessingTransformer=SimplePreProcessingTransformer,
    MaxAbsScaler=sklearn.preprocessing.MaxAbsScaler,
    SVC=sklearn.svm.SVC,
    CountVectorizer=sklearn.feature_extraction.text.CountVectorizer,
    TfidfVectorizer=sklearn.feature_extraction.text.TfidfVectorizer,
)


def get_experiment_config(file: str = EXPERIMENT_CONFIG_ALL) -> dict:
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


def save_experiment_params_as_experiment_config(file: str = EXPERIMENT_CONFIG_ALL):
    folder = file.rsplit('/', 1)[0]
    os.makedirs(folder, exist_ok=True)

    task_type_params = get_all_task_typ_params_flat()
    datasets = dataset_helper.get_all_available_dataset_names()

    with open(file, 'w') as f:
        yaml.dump(dict(
            experiment_name='All',
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


def get_all_param_grid_config_files(folder=EXPERIMENT_CONFIG_FOLDER):
    out = {}
    for file in glob('{}/*.yaml'.format(folder)):
        out[file] = get_experiment_config(file)
    return out

def get_experiment_config_for(experiment_name:str, folder=EXPERIMENT_CONFIG_FOLDER):
    file = '{}/{}.yaml'.format(folder, experiment_name)
    assert os.path.exists(file)
    return get_experiment_config(file)