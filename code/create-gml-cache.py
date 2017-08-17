#!/usr/bin/env python3

from glob import glob
import dataset_helper
import os

def main():
    args = get_args()
    print(args)
    for graph_folder in glob(args.graphs_folder + '/*'):
        if not os.path.isdir(graph_folder): continue
        print('Processing: {}'.format(graph_folder))
        graph_dataset_name = graph_folder.split('/')[-1]
        X, Y = dataset_helper.get_gml_graph_dataset(dataset_name = graph_dataset_name, graphs_folder=args.graphs_folder, cache_folder=args.cache_folder)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Convert gml graphs to cache file (X, Y)')
    parser.add_argument('--graphs_folder', type=str, help="help", default='data/graphs')
    parser.add_argument('--cache_folder', type=str, help="help", default=dataset_helper.CACHE_PATH)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()