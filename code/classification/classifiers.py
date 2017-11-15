import sklearn
from sklearn import naive_bayes

def get_classifiers():
    return [
        # sklearn.dummy.DummyClassifier(strategy='most_frequent'),
        #sklearn.naive_bayes.MultinomialNB(),
        sklearn.svm.LinearSVC(class_weight='balanced'),
        #sklearn.linear_model.PassiveAggressiveClassifier(class_weight='balanced', max_iter=args.max_iter, tol=args.tol)
        # sklearn.naive_bayes.GaussianNB(),
        # sklearn.svm.SVC(max_iter = args.max_iter, tol=args.tol),
        # sklearn.linear_model.Perceptron(class_weight='balanced', max_iter=args.max_iter, tol=args.tol),
        # sklearn.linear_model.LogisticRegression(class_weight = 'balanced', max_iter=args.max_iter, tol=args.tol),
    ]