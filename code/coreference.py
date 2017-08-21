import collections
import embeddings
import dataset_helper

def create_label_cliques_by_similarity(labels, lookup_embeddings, treshold = 0.99999, topn = 1):
    num_labels = len(labels)
    print('Creating label cliques by similarity for {} labels'.format(num_labels))
    lookup = {}
    clique_counter = 0
    cliques = collections.defaultdict(lambda: [])
    similarity_counter = {'similar': 0, 'unsimilar': 0}
    for idx, label in enumerate(labels):
        if idx % (num_labels / 10) == 0 or idx == num_labels - 1: print('Progress: {:>3}%'.format(int(100 * idx / num_labels)))
        most_similar_labels = lookup_embeddings.similar_by_word(label, topn=topn)
        for most_similar_label, similarity in most_similar_labels:
            if similarity > treshold:
                # If both labels have already been added to cliques, ...
                if label in lookup and most_similar_label in lookup:
                    # ... ignore both labels
                    continue
                # If the current label is already in the lookup, ...
                if label in lookup:
                    clique_num = lookup[label]
                    # ... add the similar label to that clique
                    cliques[clique_num].append(most_similar_label)
                    # ... and add it to the lookup
                    lookup[most_similar_label] = clique_num
                # If the similar label is already in lookup, ...
                elif most_similar_label in lookup:
                    clique_num = lookup[most_similar_label]
                    # ... add the current label to that clique
                    cliques[clique_num].append(label)
                    # ... and add it to the lookup
                    lookup[label] = clique_num
                # If neither the current label or the similar label are in the lookup, ...
                else:
                    # ... create a new clique
                    clique_num = clique_counter

                    # ... and add both to the new clique
                    cliques[clique_num].append(label)
                    cliques[clique_num].append(most_similar_label)

                    # ... add both the current label and the similar label to the lookup
                    lookup[label] = clique_num
                    lookup[most_similar_label] = clique_num

                    # ... and increment the clique counter
                    clique_counter += 1
                similarity_counter['similar'] += 1
            else:
                similarity_counter['unsimilar'] += 1
    return lookup, cliques, similarity_counter
