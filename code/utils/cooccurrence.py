import numpy as np
import spacy
from spacy.tokens.doc import Doc
from scipy.sparse import lil_matrix
from nltk.tokenize import word_tokenize


def get_coocurrence_matrix(text, window_size=2, only_forward_window=False, ignored_words=set()):
    words = _get_text_as_words(text)

    words = [word for word in words if word not in ignored_words and word.strip() != '']

    word2id, id2word = words_to_dict(words)
    num_words = len(word2id.keys())
    mat = lil_matrix((num_words, num_words), dtype=np.uint8)
    txt_len = len(words)
    for idx, word_1 in enumerate(words):
        window_size_idx = min(idx + window_size + 1, txt_len)
        words_2 = words[idx + 1:window_size_idx]
        for word_2 in words_2:
            mat[word2id[word_1], word2id[word_2]] += 1

            if not only_forward_window and word_1 != word_2:
                mat[word2id[word_2], word2id[word_1]] += 1

    return word2id, id2word, mat


def _get_text_as_words(text):
    if isinstance(text, spacy.tokens.doc.Doc):
        words = [word.text.lower().strip() for word in text]
    elif isinstance(text, str):
        words = word_tokenize(text.lower())
    elif isinstance(text, list):
        words = [word.lower().strip() if isinstance(word, str) else word.text.lower().strip() for word in text]
    else:
        raise Exception("Not a string, not a doc: '{}', type(text)=={}".format(text, type(text)))
    return words


def words_to_dict(words):
    words = set(words)
    words = sorted(list(words))
    word2id = {word: idx for idx, word in enumerate(words)}
    return word2id, {idx: word for word, idx in word2id.items()}