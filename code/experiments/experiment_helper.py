import yaml
import os
from transformers.simple_preprocessing_transformer import SimplePreProcessingTransformer
from transformers.pipelines import pipeline_helper
import experiments
from experiments import task_helper
from utils import dataset_helper, graph_metrics
import sklearn
import sklearn.preprocessing

NEEDED_FIELDS = ['experiment_name', 'params_per_type']

EXPERIMENT_CONFIG_ALL = 'configs/param_grid_configs/all.yaml'

PLACEHOLDER_LIST = dict(
    nxgraph_degrees_metric=graph_metrics.nxgraph_degrees_metric,
    nxgraph_pagerank_metric=graph_metrics.nxgraph_pagerank_metric,
    SimplePreProcessingTransformer=SimplePreProcessingTransformer(),
    MaxAbsScaler=sklearn.preprocessing.MaxAbsScaler(),
    SVC=sklearn.svm.SVC()
)

def get_experiment_config(file: str = EXPERIMENT_CONFIG_ALL) -> dict:
    assert os.path.exists(file)

    with open(file) as f:
        experiment_options = yaml.load(f.read())

    for field in NEEDED_FIELDS:
        assert field in experiment_options, 'Missing field: {}'.format(field)

    experiment_name = experiment_options['experiment_name']

    task_type_params = {}
    for task_type, params in experiment_options['params_per_type'].items():
        flattened_params = pipeline_helper.flatten_nested_params(params)
        unflattened_params = pipeline_helper.unflatten_params(flattened_params)
        assert params == unflattened_params
        task_type_params[task_type] = replace_placeholders(flattened_params)

    limit_dataset = experiment_options.get('limit_dataset', None)

    return dict(
        experiment_name=experiment_name,
        params_per_type=task_type_params,
        limit_dataset=limit_dataset
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
        if k == 'graph_gram':
            print(k, out)
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
        out_param_grid[k] = [placeholder_list[el] if el in placeholder_list else el for el in val]

    return out_param_grid
