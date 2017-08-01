import os
from glob import glob
import codecs

SPAM_LABEL = 'spam'
NOT_SPAM_LABEL = 'no_spam'

def fetch(remove_subject = True):
    X, Y = [], []
    current_folder = os.path.dirname(os.path.abspath(__file__))
    for directory in os.listdir(os.path.join(current_folder, 'src')):
        directory = os.path.join(current_folder, 'src', directory)
        if not os.path.isdir(directory): continue
        for file in glob(directory + '/bare/*/*.txt'):
            filename = file.split('/')[-1]
            is_spam = filename.startswith('spmsg')
            label = SPAM_LABEL if is_spam else NOT_SPAM_LABEL
            with codecs.open(file) as f:
                text = f.read()
                if remove_subject:
                    text = '\n'.join(text.split('\n', maxsplit = 3)[2:])
                X.append(text)
                Y.append(label)
    return X, Y

if __name__ == '__main__':
    fetch()