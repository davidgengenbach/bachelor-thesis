import sklearn
from sklearn import naive_bayes


def get_classifiers():
    return [
        # sklearn.dummy.DummyClassifier(strategy='most_frequent'),
        # sklearn.naive_bayes.MultinomialNB(),
        sklearn.svm.LinearSVC(class_weight='balanced', random_state=42),
        # sklearn.svm.SVC(class_weight='balanced'),
        # sklearn.linear_model.PassiveAggressiveClassifier(class_weight='balanced', max_iter=args.max_iter, tol=args.tol)
        # sklearn.naive_bayes.GaussianNB(),
        # sklearn.svm.SVC(max_iter = args.max_iter, tol=args.tol),
        # sklearn.linear_model.Perceptron(class_weight='balanced', max_iter=args.max_iter, tol=args.tol),
        # sklearn.linear_model.LogisticRegression(class_weight = 'balanced', max_iter=args.max_iter, tol=args.tol),
    ]


def get_classifier_params(max_iter=5000, tol=1e-4):
    return dict(
        classifier=get_classifiers(),
        classifier__C=[1e-2, 1e-1, 1],
        classifier__max_iter=[max_iter],
        classifier__tol=[tol],
        classifier__class_weight=['balanced']
    )
