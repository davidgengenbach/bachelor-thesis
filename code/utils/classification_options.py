class ClassificationOptions(object):
    def __init__(
            self,
            create_predictions: bool = True,
            force: bool = False,
            keep_coefs: bool = False,
            save_best_clf: bool = True,
            n_jobs: int = 1,
            n_jobs_outer: int = 1,
            n_splits: int = 3,
            n_splits_nested: int = 3,
            prediction_test_size: float = 0.15,
            refit: str = 'f1_macro',
            use_nested_cross_validation: bool = False,
            scoring=['precision_macro', 'recall_macro', 'accuracy', 'f1_macro'],
            results_folder: str = 'data/results',
            predictions_folder: str = 'data/results/predictions',
            classifier_folder: str = 'data/results/classifier',
            verbose: int = 11
    ):
        l = locals()
        self.create_predictions = create_predictions
        self.force = force
        self.keep_coefs = keep_coefs
        self.n_jobs = n_jobs
        self.n_jobs_outer = n_jobs_outer
        self.n_splits = n_splits
        self.prediction_test_size = prediction_test_size
        self.refit = refit
        self.scoring = scoring
        self.results_folder = results_folder
        self.predictions_folder = predictions_folder
        self.verbose = verbose
        self.save_best_clf = save_best_clf
        self.classifier_folder = classifier_folder
        self.use_nested_cross_validation = use_nested_cross_validation
        self.n_splits_nested = n_splits_nested

        # This is just to ensure that all parameters have really been assigned to 'self'
        # (to avoid copy-paste errors)
        for key, val in l.items():
            if key == 'self': continue
            assert hasattr(self, key)
            assert getattr(self, key) == val

    @classmethod
    def from_dict(cls, args: dict):
        assert isinstance(args, dict)
        opts = cls()
        for k, v in args.items():
            if hasattr(opts, k):
                setattr(opts, k, v)
        return opts

    @classmethod
    def from_argparse_options(cls, args):
        return cls.from_dict(vars(args))