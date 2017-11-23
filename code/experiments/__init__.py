from . import experiments
from .task_helper import ExperimentTask
import typing
from utils import filename_utils, graph_helper


def get_all_tasks() -> typing.List[ExperimentTask]:
    tasks = []
    for x in [experiments]:
        assert hasattr(x, 'get_tasks')
        tasks += x.get_tasks()
    tasks = sorted(tasks, key=lambda x: x.type)
    return tasks


def get_filtered_tasks(task_type=None, dataset=None, graph_type_filter:str=None, task_name_filter:str=None, tasks=None) -> typing.List[ExperimentTask]:
    task_type = _ensure_is_container(task_type)
    dataset = _ensure_is_container(dataset)

    if tasks is None:
        tasks = get_all_tasks()

    return [
        t for t in tasks if
        (not task_type or t.type in task_type) and
        (not dataset or filename_utils.get_dataset_from_filename(t.name) in dataset) and
        (not task_name_filter or task_name_filter in t.name) and
        (not graph_type_filter or graph_type_filter == graph_helper.get_graph_type_from_filename(t.name))
    ]


def _ensure_is_container(a) -> typing.List:
    if a is not None and not isinstance(a, typing.List):
        return [a]
    return a
