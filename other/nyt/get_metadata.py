#!/usr/bin/env python3

from joblib import Parallel, delayed
from glob import glob
import csv
import sys

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='desc')
    parser.add_argument('--folder', type=str, help="help", default='nyt_corpus/data/unpacked')
    parser.add_argument('--out_file', type=str, help="help", default='meta.csv')
    parser.add_argument('--n_jobs', type=int, help="help", default=10)
    parser.add_argument('--batch_size', type=int, help="help", default=10000)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    out_file = args.out_file
    folder=args.folder
    n_jobs=args.n_jobs
    batch_size=args.batch_size

    files = glob('{}/*/*/*/*.xml'.format(folder))
    
    assert len(files)

    with open(out_file, 'w') as f: pass

    batches = list(batch(files, batch_size))
    for idx, fs in enumerate(batches):
        sys.stdout.write('\r{:9}/{}'.format(idx + 1, len(batches)))
        results = Parallel(n_jobs=n_jobs)(delayed(process_file)(file) for file in fs)
        with open(out_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(results)
    print()


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_meta_content(line, attr='content', na='_'):
    els = line.split('{}="'.format(attr))
    if len(els) < 2:
        return na
    return els[1].split('"', 1)[0]

def process_file(file):
    file_name = file.split('/', 2)[-1]
    out = ['_'] * 6
    out[0] = file_name
    with open(file) as f:
        for line in f:
            if '<title>' in line:
                out[1] = line.split('>', 1)[1].split('<', 1)[0]
            elif 'print_section' in line:
                out[2] = get_meta_content(line)
            elif 'name="dsk"' in line:
                out[3] = get_meta_content(line)
            elif ' name="online_sections"' in line:
                out[4] = get_meta_content(line)
            elif 'pubdata' in line:
                out[5] = get_meta_content(line, attr='item-length')
                break
    return out


if __name__ == '__main__':
    main()