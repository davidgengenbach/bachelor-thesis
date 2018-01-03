import sklearn
import sklearn.svm

def get_classifiers():
    return [
        sklearn.svm.LinearSVC(class_weight='balanced'),
    ]


def get_classifier_params(max_iter=5000, tol=5e-4):
    return dict(
        classifier=get_classifiers(),
        classifier__C=[1e-2, 1e-1, 1],
        classifier__max_iter=[max_iter],
        classifier__tol=[tol],
        classifier__class_weight=['balanced']
    )
