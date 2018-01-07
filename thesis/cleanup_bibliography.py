#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Removes un-needed fields from a LaTex bibliography file (*.bib)'''
import re
import codecs
import collections

def main():
    args = get_args()
    import sys

    print('Processing:\n\t{}\n'.format(args.in_file))

    with codecs.open(args.in_file, errors='ignore', encoding='utf-8') as f:
        content = f.read()

    original_content = content[:]

    # Extract all entry types
    regexp_items = r'^@(.+?){'
    all_items = re.findall(regexp_items, content, re.MULTILINE)

    print('All entry types')
    print_occurrenes(all_items)
    print()

    # Extract all fields
    regexp_all_fields = r'(.+?) = \{(.*)\},'
    all_fields = re.findall(regexp_all_fields, content)
    all_field_names, field_content = zip(*all_fields)

    print('All fields:')
    print_occurrenes(all_field_names)
    print()

    # Remove fields
    print('Removing fields:')
    for field in args.remove_fields:
        regexp = r"%s = \{.*?\},\n" % (field)
        content = re.sub(regexp, '', content, flags=re.DOTALL)
        print('\t{}'.format(field))
    print()

    # Skip saving the result when nothing changed
    if original_content == content:
        print('Nothing changed. Aborting')
        return

    print('Saving to: {}'.format(args.out_file))
    with open(args.out_file, 'w') as f:
        f.write(content)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remove_fields', help='', type=str, nargs='+', default=[
        'abstract',
        'file',
        'keywords'
    ])
    # Unfortunately, TexStudio needs absolute paths...
    parser.add_argument('--in_file', type=str, help="The bibliography file to be processed", default='/Users/davidgengenbach/Documents/Projekte/bachelor-thesis/thesis/library.bib')
    parser.add_argument('--out_file', type=str, help="Path to resulting file", default='/Users/davidgengenbach/Documents/Projekte/bachelor-thesis/thesis/library.bib')
    args = parser.parse_args()
    return args

def print_occurrenes(items):
    for field, occurrences in sorted(collections.Counter(items).items(), key=lambda x: x[1]):
        print('\t{:<4}\t{}'.format(occurrences, field))

if __name__ == '__main__':
    main()