#!/usr/bin/env python3

import gensim

def main():
    args = get_args()
    all_node_labels = [x.strip() for x in open(args.in_file).read().splitlines() if x.strip() != '']
    assert len(all_node_labels)
    print('First 10 labels:', all_node_labels[:10])
    model = init_w2v_google(args.w2v_embeddings_file, args.first_line_header, args.w2v_embeddings_file_binary)
    counter = {'found': 0, 'not_found': 0}
    with open(args.out_file, 'w') as f:
        for idx, label in enumerate(all_node_labels):
            if label in model:
                embedding = model[label]
                f.write('{}\t{}'.format(label, embedding))
                counter['found'] += 1
            else:
                embedding = None
                #print('Could not retrieve embedding for: {}'.format(label))
                f.write('{}\t{}'.format(label, 'NOT_FOUND'))
                counter['not_found'] += 1
            f.write('\n')
    print('Stats:', counter)

def init_w2v_google(w2v_file, first_line_header = True, binary = False):
    if binary:
        embeddings = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=True)
    else:
        embeddings = {}
        with open(w2v_file) as f:
            if first_line_header:
                first_line = f.readline()
                num_labels, num_dim = [int(x) for x in first_line.split(' ')]
            embeddings = {x.split(' ', 1)[0].strip(): x.split(' ', 1)[1].strip() for x in f}
    return embeddings

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Generate word2vec embeddings for labels in a txt file')
    parser.add_argument('--in_file', type=str, help="The labels, one label per line", default='ling-spam-labels.txt')
    parser.add_argument('--out_file', type=str, help="The embeddings for the labels in --in_file. One per line, label and embeddings separated by a tab", default='ling-spam-embeddings.txt')
    parser.add_argument('--embeddings_float_precision', type=int, help="How many places after the comma should be saved", default=20)
    parser.add_argument('--first_line_header', action = 'store_true')
    parser.add_argument('--w2v_embeddings_file', type=str, help="The pre-trained w2v embeddings file", default='GoogleNews-vectors-negative300.bin')
    parser.add_argument('--w2v_embeddings_file_binary', action = 'store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()