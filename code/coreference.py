import collections
import embeddings
import dataset_helper
import numpy as np

def get_most_similar_labels(labels, lookup_embeddings, topn = 1):
    """Retrieves the most similar labels for a given list of labels.
    
    Args:
        labels (list(str)): the labels
        lookup_embeddings (models.keyedvectors.KeyedVectors): the embeddings that should be 
        topn (int, optional): determines how many similar labels should be returned for a given label
    
    Returns:
        dict: a dict, where the keys are the given labels and the values are a list of the most similar labels for them:
        {
            'some_label': [('similar_label', 0.999), ...],
            ...
        }
    """
    results = {}
    num_labels = len(labels)
    for idx, label in enumerate(labels):
        if idx % int(num_labels / 10) == 0 or idx == num_labels - 1: print('Progress: {:>3}%'.format(int(100 * idx / num_labels)))
        results[label] = lookup_embeddings.similar_by_word(label)
    return results

def create_label_cliques_by_similarity(similar_labels, threshold = 1 - 9e-10):
    """Returns a clustering/binning of the given labels. All labels where the similarity is greater than a given threshold are in the same clique/bin/cluster.
    
    Args:
        similar_labels (list): A list as returned by the "get_most_similar_labels" function
        threshold (float, optional): 
    
    Returns:
        dict:a dict where the keys are labels and the values the cliques-id (= a number) the labels belong to
    """
    lookup = {}
    clique_counter = 0
    for label, most_similar_labels in similar_labels.items():
        for most_similar_label, similarity in most_similar_labels:
            if similarity > threshold:
                # If both labels have already been added to cliques, ...
                if label in lookup and most_similar_label in lookup:
                    # ... ignore both labels
                    continue
                # If the current label is already in the lookup, ...
                if label in lookup:
                    clique_num = lookup[label]
                    # ... add it to the lookup
                    lookup[most_similar_label] = clique_num
                # If the similar label is already in lookup, ...
                elif most_similar_label in lookup:
                    clique_num = lookup[most_similar_label]
                    # ... and add it to the lookup
                    lookup[label] = clique_num
                # If neither the current label or the similar label are in the lookup, ...
                else:
                    # ... create a new clique
                    clique_num = clique_counter

                    # ... add both the current label and the similar label to the clique
                    lookup[label] = clique_num
                    lookup[most_similar_label] = clique_num

                    # ... and increment the clique counter
                    clique_counter += 1
    return lookup

def get_non_coreferenced_labels(labels, lookup):
    """Returns all labels that belong to no clique.
    
    Args:
        labels (list(str)): all labels
        lookup (dict): the clique lookup (keys are labels, values are the clique-numbers)
    
    Returns:
        list(str): the labels which belong to no clique
    """
    return list(set(labels) - set(lookup.keys()))