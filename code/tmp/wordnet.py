import gensim

def main():
    model = init_w2v_google('GoogleNews-vectors-negative300.bin')
    print(model['dog'])

def init_w2v_google(googlenews_vector_file = 'data/GoogleNews-vectors-negative300.bin'):
    model = gensim.models.KeyedVectors.load_word2vec_format(googlenews_vector_file, binary=True)
    return model

if __name__ == '__main__':
    main()