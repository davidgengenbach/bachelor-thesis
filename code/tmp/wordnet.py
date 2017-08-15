import gensim

def main():
    all_node_labels = [x.strip() for x in open('ling-spam-labels.txt').read().splitlines() if x.strip() != '']
    assert len(all_node_labels)
    print('First 10 labels:', all_node_labels[:10])
    model = init_w2v_google('GoogleNews-vectors-negative300.bin')
    with open('ling-spam-embeddings.txt', 'w') as f:
        for idx, label in enumerate(all_node_labels):
            try:
                embedding = model[label]
                f.write('{}\t{}'.format(label, ','.join(['{:.20f}'.format(x) for x in embedding])))
            except:
                embedding = None
                print('Could not retrieve embedding for: {}'.format(label))
                f.write(label)
            f.write('\n')

def init_w2v_google(googlenews_vector_file = 'data/GoogleNews-vectors-negative300.bin'):
    model = gensim.models.KeyedVectors.load_word2vec_format(googlenews_vector_file, binary=True)
    return model

if __name__ == '__main__':
    main()