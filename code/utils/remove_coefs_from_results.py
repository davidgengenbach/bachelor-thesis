import numpy as np
import sklearn

def remove_coefs(clf):
    found = False
    for x in ['coef_', 'class_log_prior_', 'intercept_', 'feature_log_prob_', 'class_count_', 'feature_count_', 'vocabulary_', 'idf_', 'stop_words_', '_tfidf', 'scale_', 'max_abs_']:
        try:
            setattr(clf, x, None)
            found = True
        except:
            pass

    return found

def remove_coefs_from_results(results):
    found = False
    for attr, val in results.items():
        if np.ma.isMaskedArray(val):
            val = np.ma.asarray(val)

        if not isinstance(val, list) and not isinstance(val, (np.ndarray, np.generic)) and not np.ma.isMaskedArray(val):
            val = [val]
        for clf in val:
            found |= remove_coefs(clf)
            if isinstance(clf, dict):
                for k, v in clf.items():
                    found |= remove_coefs(v)

    return found
