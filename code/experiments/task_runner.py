import os
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV
from utils import filename_utils, results_helper
from utils.classification_options import ClassificationOptions
from utils.logger import LOGGER
from utils.remove_coefs_from_results import remove_coefs_from_results
from transformers.pipelines import pipeline_helper
from . import task_helper
from .task_helper import ExperimentTask


def run_classification_task(task: ExperimentTask, cfo: ClassificationOptions, experiment_config: dict):
    args = cfo
    result_filename_tmpl = filename_utils.get_result_filename_for_task(task, experiment_config=experiment_config)
    result_file = '{}/{}'.format(cfo.results_folder, result_filename_tmpl)
    predictions_file = '{}/{}'.format(cfo.predictions_folder, result_filename_tmpl)

    if not cfo.force and os.path.exists(result_file):
        return

    X, Y, estimator, param_grid = task.fn()

    # A good heuristic of whether it's a gram matrix is whether the dimensions are the same
    is_precomputed = isinstance(X, np.ndarray) and X.shape[0] == X.shape[1]

    # This is also a heuristic
    is_dummy = 'classifier__strategy' in param_grid and 'classifier__C' not in param_grid

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

    LOGGER.info('ParamGrid: {}\n\n'.format(pipeline_helper.remove_complex_types(param_grid)))

    X_train, Y_train, X_test, Y_test, train_i, test_i = X, Y, [], [], range(len(X)), []

    if not is_dummy:  # and cfo.create_predictions:
        # Hold out validation set for predictions
        try:
            X_train, X_test, Y_train, Y_test, train_i, test_i = train_test_split(
                X,
                Y,
                test_size=cfo.prediction_test_size,
                random_state=cfo.random_state,
                is_precomputed=is_precomputed
            )
        except Exception as e:
            LOGGER.warning('Could not split dataset for predictions')
            LOGGER.exception(e)

    if cfo.n_splits == -1:
        _, _, _, _, X_train_i, X_test_i = train_test_split(X_train, Y_train, test_size=0.33, is_precomputed=is_precomputed)
        cv = [(X_train_i, X_test_i)]
    else:
        cv = sklearn.model_selection.StratifiedKFold(
            n_splits=cfo.n_splits,
            random_state=cfo.random_state,
            shuffle=True
        )

    gscv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=cfo.scoring,
        n_jobs=cfo.n_jobs,
        verbose=cfo.verbose,
        refit=cfo.refit
    )

    gscv_result = gscv.fit(X_train, Y_train)

    if not is_dummy and cfo.create_predictions:
        if not len(X_test):
            LOGGER.warning('Validation set for prediction has no items')
        else:
            try:
                # Retrain the best classifier and get prediction on validation set
                best_classifier = sklearn.base.clone(gscv_result.best_estimator_)
                best_classifier.fit(X_train, Y_train)
                Y_test_pred = best_classifier.predict(X_test)

                results_helper.save_results({
                    'results': {
                        'Y_real': Y_test,
                        'Y_pred': Y_test_pred,
                        'X_test': X_test
                    }
                }, predictions_file, args)

            except Exception as e:
                LOGGER.warning('Error while trying to retrain best classifier')
                LOGGER.exception(e)

    if not cfo.keep_coefs:
        remove_coefs_from_results(gscv_result.cv_results_)

    results_helper.save_results(gscv_result.cv_results_, result_file, args)


def train_test_split(X, Y, test_size: float = 0.15, random_state: int = 42, is_precomputed: bool = False):
    def train_test_split(*Xs, Y=None):
        return sklearn.model_selection.train_test_split(
            *Xs,
            stratify=Y,
            test_size=test_size,
            random_state=random_state
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