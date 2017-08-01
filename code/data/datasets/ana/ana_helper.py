import os
from glob import glob
import codecs

PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(PATH_TO_HERE, 'src')

def get_ana_set(dataset_name, preprocessing_type = 'all-terms', sets = ('train', 'test'), dataset_folder = DATASET_FOLDER):
    if preprocessing_type:
        preprocessing_type = '-' + preprocessing_type
    else:
        preprocessing_type = ''
    filenames = [os.path.join(dataset_folder, '{}-{}{}.txt'.format(dataset_name, s, preprocessing_type)) for s in sets]
    X, Y = [], []
    for file in filenames:
        data = get_topic_file(file)
        Y += [topic for topic, text in data]
        X += [text for topic, text in data]
    return X, Y

def get_topic_file(file):
    """Reads in a file of documents. Each line corresponds to a document.
    On each line, the first part is the topic of that document. The doc follows after a "\t".

    Args:
        file (str): The file
    
    Returns:
        list: returns a list of lists. One row: list('topic', 'word1 word2')
    """
    with codecs.open(file, 'r') as f:
        data = f.read().split('\n')
        data = [x.strip().split('\t') for x in data if x.strip() != '' and len(x.strip().split('\t')) == 2]
        return data