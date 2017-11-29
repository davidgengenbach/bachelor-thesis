#!/usr/bin/env python3

from glob import glob
import os

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='desc')
    parser.add_argument('--pattern', type=str, help="help", default='*.txt')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    files = [x for x in glob(args.pattern) if 'w2v' not in x]
    for file in files:
        print('Converting: {}'.format(file))
        out_file = file.replace('.txt', '.w2v.txt')
        if not args.force and os.path.exists(out_file):
             continue
        with open(file) as f:
            first_item = f.readline()
            dimension = first_item.count(' ')
            f.seek(0)
            assert str(dimension) in file

            for i, _ in enumerate(f): pass
            num_items = i + 1
            f.seek(0)
            with open(out_file, 'w') as f_out:
                f_out.write('{} {}\n'.format(num_items, dimension))
                for line in f:
                    f_out.write(line)


if __name__ == '__main__':
    main()