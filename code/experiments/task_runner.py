import os
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV
from utils import filename_utils, results_helper, helper, constants
from utils.classification_options import ClassificationOptions
from utils.logger import LOGGER
from utils.remove_coefs_from_results import remove_coefs_from_results
from transformers.pipelines import pipeline_helper
from . import task_helper
from . import experiment_helper
from .task_helper import ExperimentTask
from time import time


def run_classification_task(task: ExperimentTask, cfo: ClassificationOptions, experiment_config: dict):
    helper.set_random_seed()

    args = cfo
    result_filename_tmpl = filename_utils.get_result_filename_for_task(task, experiment_config=experiment_config, cfo=cfo)

    result_file = '{}/{}'.format(cfo.results_folder, result_filename_tmpl)
    predictions_file = '{}/{}'.format(cfo.predictions_folder, result_filename_tmpl)
    classifier_file = '{}/{}'.format(cfo.classifier_folder, result_filename_tmpl)

    if not cfo.force and os.path.exists(result_file):
        return

    time_checkpoints = {}

    def add_time_checkpoint(name):
        time_checkpoints[name] = time()

    add_time_checkpoint('start')
    X, Y, estimator, param_grid = task.fn()
    add_time_checkpoint('retrieved_data')

    # A good heuristic of whether it's a gram matrix is whether the dimensions are the same
    is_precomputed = isinstance(X, np.ndarray) and X.shape[0] == X.shape[1]

    # This is also a heuristic
    is_dummy = 'classifier__strategy' in param_grid

    # Add classifiers, instantiate transformer classes and merge with experiment config
    param_grid = experiment_helper.prepare_param_grid(task, param_grid, experiment_config)

    LOGGER.info('ParamGrid: {}\n\n'.format(pipeline_helper.remove_complex_types(param_grid)))

    X_train, Y_train, X_test, Y_test, train_i, test_i = X, Y, [], [], range(len(X)), []

    if not is_dummy:  # and cfo.create_predictions:
        # Hold out validation set for predictions
        try:
            X_train, X_test, Y_train, Y_test, train_i, test_i = train_test_split(
                X,
                Y,
                test_size=cfo.prediction_test_size,
                is_precomputed=is_precomputed,
            )
        except Exception as e:
            LOGGER.warning('Could not split dataset for predictions')
            LOGGER.exception(e)

    def get_cv(splits):
        if splits == -1:
            _, _, _, _, X_train_i, X_test_i = train_test_split(X_train, Y_train, test_size=0.33, is_precomputed=is_precomputed)
            cv = [(X_train_i, X_test_i)]
        else:
            cv = sklearn.model_selection.StratifiedKFold(
                n_splits=cfo.n_splits,
                shuffle=True,
                random_state=constants.RANDOM_SEED
            )
        return cv

    add_time_checkpoint('split_data')
    cv = get_cv(cfo.n_splits)

    should_refit = np.all([
        #not cfo.use_nested_cross_validation,
        not is_dummy,
        #cfo.create_predictions or cfo.save_best_clf
    ])

    gscv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=cfo.scoring,
        n_jobs=cfo.n_jobs,
        verbose=cfo.verbose,
        refit=cfo.refit if should_refit else False
    )

    if cfo.use_nested_cross_validation and not is_dummy:
        cv_nested = get_cv(cfo.n_splits_nested)

        LOGGER.info('Using nested cross-validation')

        scores = sklearn.model_selection.cross_validate(gscv, X, Y, scoring=cfo.scoring, cv=cv_nested, n_jobs=cfo.n_jobs_outer, verbose=cfo.verbose, return_train_score=True)
        result = dict(scores, **param_grid)
        add_time_checkpoint('fitted_nested')
        results_helper.save_results(result, result_file, args, time_checkpoints=time_checkpoints)
        return

    gscv_result = gscv.fit(X_train, Y_train)
    add_time_checkpoint('fitted_gridsearch')

    if not is_dummy and cfo.create_predictions:
        if not len(X_test):
            LOGGER.warning('Validation set for prediction has no items')
        else:
            try:
                # Retrain the best classifier and get prediction on validation set
                Y_test_pred = gscv_result.best_estimator_.predict(X_test)
                add_time_checkpoint('predicted')
                results_helper.save_results({
                    'gscv_result': remove_coefs_from_results(gscv_result.cv_results_),
                    'all_params': remove_coefs_from_results(param_grid),
                    'best_params': remove_coefs_from_results(gscv_result.best_params_),
                    'Y_real': Y_test,
                    'Y_pred': Y_test_pred,
                    'X_test': X_test,
                }, predictions_file, args, time_checkpoints=time_checkpoints)
            except Exception as e:
                LOGGER.warning('Error while trying to retrain best classifier')
                LOGGER.exception(e)

    if cfo.save_best_clf:
        best_estimator = gscv_result.best_estimator_
        try:
            results_helper.save_results({
                'params': gscv_result.best_params_,
                'classifier': best_estimator
            }, classifier_file, args, time_checkpoints=time_checkpoints)
        except Exception as e:
            LOGGER.warning('Error while saving best estimator: {}'.format(e))
            LOGGER.exception(e)

    add_time_checkpoint('finished')
    results_helper.save_results(gscv_result.cv_results_, result_file, args, time_checkpoints=time_checkpoints)


def train_test_split(X, Y, test_size: float = 0.15, is_precomputed: bool = False):
    def train_test_split(*Xs, Y=None):
        return sklearn.model_selection.train_test_split(
            *Xs,
            stratify=Y,
            test_size=test_size,
            random_state=constants.RANDOM_SEED
        )

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
        indices = list(range(len(X)))
        X_train, X_test, Y_train, Y_test, X_train_i, X_test_i = train_test_split(X, Y, indices, Y=Y)

    return X_train, X_test, Y_train, Y_test, X_train_i, X_test_i


def get_precomputed_subset(K, indices1, indices2=None):
    if not indices2:
        indices2 = indices1
    indices = np.meshgrid(indices1, indices2, indexing='ij')
    return np.array(K)[indices]