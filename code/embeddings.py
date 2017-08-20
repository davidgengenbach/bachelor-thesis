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