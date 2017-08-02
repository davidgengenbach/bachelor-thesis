import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from time import time

def preprocess_text(text, remove_stopwords_ = False, remove_interpunction_ = False):
    reg = r'\n\n+'
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

def remove_stopwords(tokens, stopwords = set(stopwords.words('english'))):
    return [i for i in tokens if i not in stopwords]

def remove_interpunction(tokens):
    return [w.lower() for w in tokens if w.isalnum()]
