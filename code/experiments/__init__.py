from . import experiments
from .task_helper import ExperimentTask

def get_all_tasks():
    tasks = []
    for x in [experiments]:
        assert hasattr(x, 'get_tasks')
        tasks += x.get_tasks()

    tasks = sorted(tasks, key=lambda x: x.type)
    return tasks