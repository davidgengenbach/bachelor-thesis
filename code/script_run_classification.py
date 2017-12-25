#!/usr/bin/env python

import os
import collections
import configargparse as argparse
import typing
from time import time
import numpy as np
import gc

from utils import graph_helper
from utils.logger import LOGGER
from utils import filename_utils, time_utils, helper
from utils.classification_options import ClassificationOptions

import experiments
from experiments.task_helper import ExperimentTask
from experiments import task_runner, experiment_helper


def get_args():
    parser = argparse.ArgumentParser(
        description='Run classification on the text and graph datasets',
        default_config_files=['configs/run_classification.yaml']
    )

    parser.add_argument('--config', is_config_file=True)
    parser.add_argument('--experiment_config', type=str, default=None)

    # Options
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--dry_run', action='store_true', help='Do not classify. Only show description, tasks, ...')
    parser.add_argument('--results_folder', type=str, default='data/results')
    parser.add_argument('--predictions_folder', type=str, default='data/results/predictions')
    parser.add_argument('--classifier_folder', type=str, default='data/results/classifier')
    parser.add_argument('--force', action='store_true', help='Overwrite already calculated results')


    # Options for Classifier and CV
    parser.add_argument('--scoring', type=str, nargs='+', default=['precision_macro', 'recall_macro', 'accuracy', 'f1_macro'])
    parser.add_argument('--refit', type=str, default='f1_macro')
    parser.add_argument('--use_nested_cross_validation', type=helper.argparse_str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--n_splits', type=int, default=3)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--create_predictions', type=helper.argparse_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--save_best_clf', type=helper.argparse_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--prediction_test_size', type=float, default=0.15)
    parser.add_argument('--keep_coefs', action='store_true')

    # Task filters
    parser.add_argument('--task_name_filter', type=str, default=None)
    parser.add_argument('--task_type_include_filter', type=str, nargs='+', default=None)
    parser.add_argument('--task_type_exclude_filter', type=str, nargs='+', default=None)
    parser.add_argument('--limit_dataset', nargs='+', type=str, default=None)

    args = parser.parse_args()
    return args



def main():
    args = get_args()

    helper.print_script_args_and_info(args)

    if args.experiment_config:
        experiment_config = experiment_helper.get_experiment_config(args.experiment_config)
    else:
        experiment_config = {}

    create_results_dir(args)

    classification_options: ClassificationOptions = ClassificationOptions.from_argparse_options(args)

    tasks: typing.List[ExperimentTask] = experiments.get_all_tasks()

    start_tasks(args, tasks, classification_options, experiment_config)


def start_tasks(args, all_tasks: typing.List[ExperimentTask], classification_options: ClassificationOptions, experiment_config: dict):
    filtered_task_types = experiment_config['params_per_type'].keys() if experiment_config else None

    if experiment_config.get('limit_dataset', None) is not None:
        limit_dataset = experiment_config['limit_dataset']
    else:
        limit_dataset = args.limit_dataset

    limit_graph_type = experiment_config.get('limit_graph_type', None)

    def should_process_task(task: ExperimentTask):
        # Dataset filter
        is_filtered_by_dataset = limit_dataset and filename_utils.get_dataset_from_filename(task.name) not in limit_dataset

        # Task type filters
        is_filtered_by_include_filter = (args.task_type_include_filter and task.type not in args.task_type_include_filter)
        is_filtered_by_exclude_filter = (args.task_type_exclude_filter and task.type in args.task_type_exclude_filter)

        is_filtered_by_name_filter = (args.task_name_filter and args.task_name_filter not in task.name)
        is_filtered_by_param_options = (filtered_task_types and task.type not in filtered_task_types)
        is_filtered_by_graph_type = (limit_graph_type and graph_helper.get_graph_type_from_filename(task.name) not in [None] + limit_graph_type)


        # Do not process tasks that have already been calculated (unless args.force == True)
        created_files = ['{}/{}'.format(args.results_folder, filename_utils.get_result_filename_for_task(task, experiment_config, cfo=classification_options))]
        is_filtered_by_file_exists = (not args.force and np.any([os.path.exists(file) for file in created_files]))

        should_process = not np.any([
            is_filtered_by_graph_type,
            is_filtered_by_dataset,
            is_filtered_by_include_filter,
            is_filtered_by_name_filter,
            is_filtered_by_file_exists,
            is_filtered_by_exclude_filter,
            is_filtered_by_param_options
        ])

        return should_process

    def print_tasks(tasks: typing.List[ExperimentTask]):
        for task in tasks:
            print('\t{t.type:26} {dataset:18} {t.name}'.format(t=task, dataset = filename_utils.get_dataset_from_filename(task.name)))
        print('\n')

    # Filter out tasks
    tasks = sorted([task for task in all_tasks if should_process_task(task)], key = lambda x: filename_utils.get_dataset_from_filename(x.name))

    if args.dry_run:
        print('All tasks:')
        print_tasks(all_tasks)

    print('Filtered tasks:')
    print_tasks(tasks)

    print('# tasks per type (filtered/unfiltered)')
    task_type_counter_unfiltered = collections.Counter([t.type for t in all_tasks])
    task_type_counter_filtered = collections.Counter([t.type for t in tasks])
    for task_type, unfiltered_count in task_type_counter_unfiltered.items():
        print('\t{:25} {:2}/{:2}'.format(task_type, task_type_counter_filtered.get(task_type, 0), unfiltered_count ))
    print('\n')

    if args.dry_run:
        print('Only doing a dry-run. Exiting.')
        return

    num_tasks = len(tasks)
    for task_idx, t in enumerate(tasks):
        def print_task(msg = ''):
            LOGGER.info('Task {idx:>2}/{num_tasks}: {t.type:30} - {t.name:40} - {msg}'.format(
                idx = task_idx + 1,
                num_tasks = num_tasks,
                t = t,
                msg = msg
            ))

        start_time = time()
        print_task('Started')
        try:
            task_runner.run_classification_task(t, classification_options, experiment_config)
            gc.collect()
        except Exception as e:
            print_task('Error: {}'.format(e))
            LOGGER.exception(e)
        elapsed_seconds = time() - start_time
        print_task('Finished (time={})'.format(time_utils.seconds_to_human_readable(elapsed_seconds)))
        gc.collect()

    LOGGER.info('Finished!')


def create_results_dir(args):
    for folder in [args.classifier_folder, args.results_folder, args.predictions_folder]:
        os.makedirs(folder, exist_ok=True)


if __name__ == '__main__':
    main()
