#!/usr/bin/env python3
#!/usr/bin/env python3

from glob import glob
import os
import sys
import shutil

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='desc')
    parser.add_argument('--folder', type=str, default='nyt_corpus/data/unpacked/')
    parser.add_argument('--out_folder', type=str, default='filtered_articles')
    parser.add_argument('--file_list', type=str, default='filtered_articles.txt')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    file_list = args.file_list
    folder = args.folder
    out_folder = args.out_folder

    os.makedirs(out_folder, exist_ok=True)

    files = get_xml_files(file_list, prefix=folder)
    
    assert len(files)

    for idx, file in enumerate(files):
        sys.stdout.write('\r{:9}/{}'.format(idx + 1, len(files)))
        if not os.path.exists(file):
            raise FileNotFoundError(file)
        new_filename = '/'.join([out_folder] + file.rsplit('/', 4)[1:])
        new_folder = new_filename.rsplit('/', 1)[0]
        os.makedirs(new_folder, exist_ok=True)
        shutil.copy(file, new_filename)
        with open(file) as f:
            assert '<block class="full_text">' in f.read(), 'No full_text found in file: {}'.format(file)
    print()

def get_xml_files(file_list: str = 'filtered_articles.txt', prefix: str='data/unpacked/'):
    if not os.path.exists(file_list): 
        raise FileNotFoundError(file_list)
    with open(file_list) as f:
        files = [prefix + x.strip() for x in f.readlines() if x.strip() != '']
    return files

if __name__ == '__main__':
    main()