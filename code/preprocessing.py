import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from time import time
import spacy
import re

LINEBREAK_REGEX = re.compile(r'((\r\n)|[\n\v])+')
NONBREAKING_SPACE_REGEX = re.compile(r'(?!\n)\s+')

nlp = None

def init_spacy():
    global nlp
    if nlp: return
    nlp = spacy.load('en')

def get_spacy_parse(texts, batch_size = 100, n_threads = 1):
    init_spacy()
    return nlp.pipe(texts, batch_size=batch_size, n_threads=n_threads)


def normalize_whitespace(text):
    """
    Given ``text`` str, replace one or more spacings with a single space, and one
    or more linebreaks with a single newline. Also strip leading/trailing whitespace.
    Taken from http://textacy.readthedocs.io/en/latest/_modules/textacy/preprocess.html
    """
    return NONBREAKING_SPACE_REGEX.sub(' ', LINEBREAK_REGEX.sub(r'\n', text)).strip()

def ana_preprocessing(text):
    # all lowercase
    text = text.lower()
    # replace TAB, NEWLINE and RETURN characters by SPACE
    replacements = (
        ('\t', ' '),
        ('\n', ' '),
        ('\r', ' '),
    )
    for d in replacements:
        text = text.replace(*d)

    text = normalize_whitespace(text)
    return text


def preprocess_text_spacy(texts, min_length=-1, concat=True, n_jobs=2, batch_size=100, only_nouns = True, remove_whitespace = True, ana_preprocessing_ = True):
    """Preprocesses text by
    - only keeping the NOUNs
    - only keeping the words that are longer than min_length (optional)

    Args:
        texts (list of str): list of texts
        min_length (int, optional): the minimum length a word must have (-1 means disabling this filter)
        n_jobs (int, optional): how many threads should be used
        batch_size (int, optional): how many texts should be done per thread

    Returns:
        list of str: the pre-processed text
    """
    if ana_preprocessing_:
        texts = [ana_preprocessing(text) for text in texts]

    res = [
        [
            word for word in doc if (not remove_whitespace or word.text.strip() != '') and (not only_nouns or word.pos_ == 'NOUN') and (min_length == -1 or len(word.text) > min_length)
        ]
        for doc in get_spacy_parse(texts, batch_size=batch_size, n_threads=n_jobs)
    ]
    if concat:
        return [" ".join([word.text for word in doc] for doc in res)]
    else:
        return res


def preprocess_text_old(text):
    reg = r'\n\n+'
    text = '\n'.join(x.strip() for x in text.split('\n') if x.strip() != '')
    text = re.sub('\n', ' ', re.sub(reg, '//', text)).replace('//', '\n')
    return text


def preprocess_text(text, lower=True, remove_stopwords_=False, remove_interpunction_=False):
    reg = r'\n\n+'
    if lower:
        text = text.lower()
    text = '\n'.join(x.strip() for x in text.split('\n') if x.strip() != '')
    if remove_stopwords_ or remove_interpunction_:
        tokens = to_tokens(text)
        if remove_stopwords_:
            tokens = remove_stopwords(tokens)
        if remove_interpunction_:
            tokens = remove_interpunction(tokens)
        text = " ".join(tokens)
    return text


def to_tokens(text):
    return word_tokenize(text.lower())


def remove_stopwords(tokens, stopwords=set(stopwords.words('english'))):
    return [i for i in tokens if i not in stopwords]


def remove_interpunction(tokens):
    return [w.lower() for w in tokens if w.isalnum()]
