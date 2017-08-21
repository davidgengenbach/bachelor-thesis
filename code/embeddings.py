def get_embedding_model(w2v_file, binary = False, first_line_header = True, with_gensim = False):
    import gensim
    if binary or with_gensim:
        embeddings = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=binary)
    else:
        embeddings = {}
        with open(w2v_file) as f:
            if first_line_header:
                first_line = f.readline()
                num_labels, num_dim = [int(x) for x in first_line.split(' ')]
            embeddings = {x.split(' ', 1)[0].strip(): x.split(' ', 1)[1].strip() for x in f}
    return embeddings

def get_embeddings_for_labels(labels, embedding, check_most_similar = False, restrict_vocab = None, lookup_embedding = None, topn = 20):
    """Returns the embeddings for given labels. Optionally also tries to resolve missing embeddings with another embedding, the lookup_embedding, by looking for the most_similar words that are also in the original embedding.
    
    Args:
        labels (list(str)): A list of strings
        embedding (gensim.models.keyedvectors.KeyedVectors): the embedding to search the labels in
        check_most_similar (bool, optional): Whether to try to try to find a similar word in another embedding, the lookup_embedding
        restrict_vocab (list(str), optional): The labels that can be looked up in the lookup_embedding - these are most likely the labels that are in both the embedding and lookup_embedding
        lookup_embedding (gensim.models.keyedvectors.KeyedVectors, optional): The backup embedding to search similar words that are also in the original embedding
        topn (int, optional): How many most similar words to retrieve from the lookup embedding
    
    Returns:
        tuple(dict, list(str)): the dictionary with the labels as keys and the embeddings for that labels as value. And a list of label that could not be found
    """
    assert not check_most_similar or lookup_embedding is not None
    not_found, embeddings = [], {}

    for label in labels:
        label = label.lower()
        if label in embedding:
            embeddings[label] = embedding[label]
        elif check_most_similar and label in lookup_embedding:
            most_similar = lookup_embedding.similar_by_word(label, topn = topn)
            most_similar_labels = [label for label, similarity in most_similar]
            match = set(most_similar_labels) & set(restrict_vocab)

            if len(match):
                most_similar_label_found = [label for label, similarity in most_similar if label in match][0]
                embeddings[label] = embedding[most_similar_label_found]
            else:
                not_found.append(label)
        else:
            not_found.append(label)
    return embeddings, not_found


def get_embeddings_for_labels_with_lookup(all_labels, trained_embedding, pre_trained_embedding):
    embeddings_trained_labels = set(trained_embedding.vocab.keys())
    not_found_trained = set(all_labels) - embeddings_trained_labels

    embeddings_pre_trained_labels = set(pre_trained_embedding.vocab.keys())
    not_found_pre_trained = set(all_labels) - embeddings_pre_trained_labels

    in_both = embeddings_trained_labels & embeddings_pre_trained_labels

    embeddings_pre_trained, not_found_pre_trained_coreferenced = get_embeddings_for_labels(all_labels, pre_trained_embedding, check_most_similar = True, restrict_vocab = in_both, lookup_embedding = trained_embedding)

    return embeddings_pre_trained, not_found_pre_trained_coreferenced, not_found_trained, not_found_pre_trained


def save_embedding_dict(embedding, filename):
    num_embeddings = len(embedding.keys())
    num_dim = len(embedding[list(embedding.keys())[0]])
    with open(filename, 'w') as f:
        f.write('{} {}\n'.format(num_embeddings, num_dim))
        f.write("\n".join(['{} {}'.format(label, ' '.join([str(x) for x in vec])) for label, vec in embedding.items() if label.strip() != '']))