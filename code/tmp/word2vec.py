#!/usr/bin/env python3

def main():
    args = get_args()
    all_node_labels = [x.strip() for x in open(args.in_file).read().splitlines() if x.strip() != '']
    assert len(all_node_labels)
    print('First 10 labels:', all_node_labels[:10])
    model = init_w2v_google(args.w2v_embeddings_file)
    counter = {'found': 0, 'not_found': 0}
    with open(args.out_file, 'w') as f:
        for idx, label in enumerate(all_node_labels):
            try:
                embedding = model[label]
                f.write('{}\t{}'.format(label, ','.join(['{:.' + args.embeddings_float_precision + '}'.format(x) for x in embedding])))
                counter['found'] += 1
            except:
                embedding = None
                print('Could not retrieve embedding for: {}'.format(label))
                f.write('{}\t{}'.format(label, 'NOT_FOUND'))
                counter['not_found'] += 1
            f.write('\n')
    print('Stats:', counter)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Generate word2vec embeddings for labels in a txt file')
    parser.add_argument('--in_file', type=str, help="The labels, one label per line", default='ling-spam-labels.txt')
    parser.add_argument('--out_file', type=str, help="The embeddings for the labels in --in_file. One per line, label and embeddings separated by a tab", default='ling-spam-embeddings.txt')
    parser.add_argument('--embeddings_float_precision', type=int, help="How many places after the comma should be saved", default=20)
    parser.add_argument('--w2v_embeddings_file', type=str, help="The pre-trained w2v embeddings file", default='GoogleNews-vectors-negative300.bin')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()