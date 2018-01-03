#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import codecs

def main():
    args = get_args()
    import sys

    with codecs.open(args.in_file, errors='ignore') as f:
        content = f.read()

    original_content = content[:]

    regexp_all_fields = r'(.+?) = \{(.*)\},'
    all_fields = re.findall(regexp_all_fields, content)
    all_field_names, field_content = zip(*all_fields)

    print('All fields in {}:'.format(args.in_file))
    for field in sorted(set(all_field_names)):
        print('\t{}'.format(field))
    print()
    print('Removing fields:')
    for field in args.remove_fields:
        regexp = r"%s = \{.*?\},\n" % (field)
        content = re.sub(regexp, '', content, flags=re.DOTALL)
        print('\t{}'.format(field))
    print()

    if original_content == content:
        print('Nothing changed. Aborting')
        return

    print('Saving to: {}'.format(args.out_file))
    with open(args.out_file, 'w') as f:
        f.write(content)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Cleans up the Bib extracted from Mendeley Desktop')
    parser.add_argument('--in_file', type=str, help="help", default='/Users/davidgengenbach/Documents/Projekte/bachelor-thesis/thesis/library.bib')
    parser.add_argument('--out_file', type=str, help="help", default='/Users/davidgengenbach/Documents/Projekte/bachelor-thesis/thesis/library.bib')

    parser.add_argument('--remove_fields', type=str, nargs='+', default=[
        'abstract',
        'file',
        'keywords'
    ])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()