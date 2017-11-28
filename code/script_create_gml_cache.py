#!/usr/bin/env python3

import os
import traceback
from glob import glob

from utils import dataset_helper, helper


def main():
    args = get_args()
    helper.print_script_args_and_info(args)

    for graph_folder in glob(args.graphs_folder + '/*'):
        if graph_folder.rsplit('/', 1)[1].startswith('_'): continue
        if not os.path.isdir(graph_folder): continue
        print('Processing: {}'.format(graph_folder))
        graph_dataset_name = graph_folder.split('/')[-1]
        try:
            X, Y = dataset_helper.get_gml_graph_dataset(
                dataset_name=graph_dataset_name,
                graphs_folder=args.graphs_folder,
                use_cached=args.use_cached
            )
        except Exception as e:
            traceback.print_exc()
            print('Error:', e)


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Convert gml graphs to cache file (X, Y)')
    parser.add_argument('--graphs_folder', type=str, help="help", default='data/graphs')
    parser.add_argument('--cache_folder', type=str, help="help", default=dataset_helper.CACHE_PATH)
    parser.add_argument('--use_cached', help="help", action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
