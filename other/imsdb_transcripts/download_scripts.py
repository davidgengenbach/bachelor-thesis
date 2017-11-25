#!/usr/bin/env python3

import pandas as pd
import os
from joblib import Parallel, delayed
import requests

def main():
    args = get_args()

    os.makedirs(args.html_folder, exist_ok=True)
    df_scripts = pd.read_json(args.scripts_json)
    links = df_scripts.link.values

    Parallel(n_jobs=args.n_jobs)(delayed(download_script)(args.html_folder, link) for link in links)
    
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Download imsdb scripts')
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument('--html_folder', type=str, default='data/html/')
    parser.add_argument('--scripts_json', type=str, default='scripts.json')
    args = parser.parse_args()
    return args


def download_script(folder, link):
    filename=folder + "".join(x for x in link if x.isalnum())
    if os.path.exists(filename): return
    try:
        res = requests.get(link)
        with open(filename, 'w') as f:
            f.write('{}\n\n{}'.format(link, res.text))
    except Exception as e:
        print('Error while fetching: {}\n\n{}'.format(link, e))
        

if __name__ == '__main__':
    main()
