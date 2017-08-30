import collections
import embeddings
import dataset_helper
import numpy as np
import pandas as pd



def get_most_similar_labels(labels, lookup_embeddings, topn=1):
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
        if (num_labels >= 10 and idx % int(num_labels / 10) == 0) or idx == num_labels - 1:
            print('Progress: {:>3}%'.format(int(100 * idx / num_labels)))
        if label in lookup_embeddings:
            results[label] = lookup_embeddings.similar_by_word(label, topn = topn)
    return results


def create_label_cliques_by_similarity(similar_labels, threshold=1 - 9e-10, lookup=None, topn=-1):
    """Returns a clustering/binning of the given labels. All labels where the similarity is greater than a given threshold are in the same clique/bin/cluster.

    Args:
        similar_labels (list): A list as returned by the "get_most_similar_labels" function
        threshold (float, optional): 

    Returns:
        dict:a dict where the keys are labels and the values the cliques-id (= a number) the labels belong to
    """
    if lookup is None:
        lookup = {}
    clique_counter = 0
    for label, most_similar_labels in similar_labels.items():
        most_similar_labels = sorted(most_similar_labels, key=lambda x: x[1], reverse=True)

        if topn != -1:
            most_similar_labels = most_similar_labels[:topn]

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


def get_cliques_from_lookup(lookup):
    cliques = collections.defaultdict(lambda: [])
    for label, clique_num in lookup.items():
        cliques[clique_num].append(label)
    return cliques


def get_non_coreferenced_labels(labels, lookup):
    """Returns all labels that belong to no clique.

    Args:
        labels (list(str)): all labels
        lookup (dict): the clique lookup (keys are labels, values are the clique-numbers)

    Returns:
        list(str): the labels which belong to no clique
    """
    return list(set(labels) - set(lookup.keys()))


def plot_lookup_histogram(lookup, num_labels, title = None, figsize = (14, 6), dpi = 120):
    import matplotlib.pyplot as plt
    cliques = get_cliques_from_lookup(lookup)
    similarity_counter = {'merged': len(lookup.keys()), 'unmerged': num_labels - len(lookup.keys())}
    clique_lenghts = [len(x) for x in list(cliques.values())]
    fig, axes = plt.subplots(1, 2, figsize = figsize, dpi = dpi)

    pd.DataFrame(clique_lenghts).plot(ax = axes[0], kind = 'hist', logy = True, bins = 100, cumulative=True, normed=1, legend = False, title = "Histogram of clique lengths (cumulative)")
    pd.DataFrame(list(similarity_counter.items()), columns = ['name', 'count']).set_index('name').plot(ax = axes[1], kind = 'bar', legend = False, title = '# of labels that have been merged vs. not merged')

    if title:
        fig.suptitle(title, fontsize = 16)
    
    fig.tight_layout()

    if title:
        fig.subplots_adjust(top=0.85)

    return fig, axes