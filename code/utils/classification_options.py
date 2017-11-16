class ClassificationOptions(object):
    def __init__(
            self,
            create_predictions: bool = True,
            force: bool = False,
            keep_coefs: bool = False,
            n_jobs: int = 1,
            n_splits: int = 3,
            prediction_test_size: float = 0.15,
            random_state: int = 42,
            refit: str = 'f1_macro',
            scoring=['precision_macro', 'recall_macro', 'accuracy', 'f1_macro'],
            results_folder: str = 'data/results',
            predictions_folder: str = 'data/results/predictions',
            verbose: int = 11
    ):
        l = locals()
        self.create_predictions = create_predictions
        self.force = force
        self.keep_coefs = keep_coefs
        self.n_jobs = n_jobs
        self.n_splits = n_splits
        self.prediction_test_size = prediction_test_size
        self.random_state = random_state
        self.refit = refit
        self.scoring = scoring
        self.results_folder = results_folder
        self.predictions_folder = predictions_folder
        self.verbose = verbose

        # This is just to ensure that all parameters have really been assigned to 'self'
        # (to avoid copy-paste errors)
        for key, val in l.items():
            if key == 'self':
                continue

            assert hasattr(self, key)
            assert getattr(self, key) == val
