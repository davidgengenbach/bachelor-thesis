import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from time import time
import spacy

nlp = spacy.load('en')
nlp.pipeline = [nlp.tagger]

def get_spacy_parse(texts, batch_size = 100, n_threads = 1):
    return nlp.pipe(texts, batch_size=batch_size, n_threads=n_threads)

def preprocess_text_spacy(texts, min_length=-1, concat=True, n_jobs=2, batch_size=100, only_nouns = True):
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
    res = [[word for word in doc if (not only_nouns or word.pos_ == 'NOUN') and (min_length == -1 or len(word.text) > min_length)] for doc in get_spacy_parse(texts, batch_size=batch_size, n_threads=n_jobs)]
    if concat:
        return [" ".join([word.text for word in doc] for doc in res)]
    else:
        return res


def preprocess_text_(text, min_length=-1):
    doc = nlp(text)
    return " ".join(word.text for word in doc if word.pos_ == 'NOUN' and (min_length != -1 or len(word.text) > min_length)).lower()


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
