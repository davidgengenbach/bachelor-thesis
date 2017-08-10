import numpy as np
import spacy
from spacy.tokens.doc import Doc
from scipy.sparse import lil_matrix
from nltk.tokenize import word_tokenize, sent_tokenize
import networkx as nx
import collections

def words_to_dict(words, remove_point = True):
    words = set(words)
    words = sorted(list(words))
    word2id = {word: idx for idx, word in enumerate(words)}
    return word2id, {idx: word for word, idx in word2id.items()}

def get_coocurrence_matrix(text, window_size = 2, ignore_sentence_bounds = True, only_forward_window = False):
    # TODO
    if isinstance(text, spacy.tokens.doc.Doc):
        words = [word.text for word in text if not ignore_sentence_bounds or word.pos != spacy.symbols.PUNCT]
    elif isinstance(text, str):
        text = text.replace('.', ' ' if ignore_sentence_bounds else ' . ').lower()
        words = word_tokenize(text)
    elif isinstance(text, list):
        words = words = [word if isinstance(word, str) else word.text for word in text]
    else:
        assert False,"Not a string, not a doc: '{}', type(text)=={}".format(text, type(text))
    word2id, id2word = words_to_dict(words)
    num_words = len(word2id.keys())
    mat = lil_matrix((num_words, num_words), dtype=np.int8)
    txt_len = len(words)
    for idx, word_1 in enumerate(words):
        if word_1 == '.':
            continue
        window_size_idx = min(idx + window_size + 1, txt_len)
        words_2 = words[idx + 1:window_size_idx]
        for word_2 in words_2:
            if not ignore_sentence_bounds and word_2 == '.':
                break
            mat[word2id[word_1], word2id[word_2]] += 1

            if not only_forward_window and word_1 != word_2:
                mat[word2id[word_2], word2id[word_1]] += 1
    return word2id, id2word, mat

def plot_cooccurrence_matrix(id2word, mat, ax = None):
    if not ax:
        fig = plt.figure()
        ax = plt.gca()
    graph = nx.from_numpy_matrix(mat.toarray())
    nx.relabel_nodes(graph, id2word, copy = False)
    pos = nx.circular_layout(graph)
    edge_labels = [(source, target, data['weight']) for source, target, data in graph.edges(data = True)]
    labels = nx.get_edge_attributes(graph,'weight')
    nx.draw_networkx_edge_labels(graph,pos,edge_labels=labels, hold = False)
    nx.draw_networkx(graph, pos = pos, node_size= 900, node_color='white', hold = False, width = [weight for source, target, weight in edge_labels])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def print_cooccurrence_matrix(id2word, mat):
    words = [x[1] for x in sorted(id2word.items(), key = lambda x: x[0])]
    mat = mat.toarray()
    print('\t{}'.format(",".join(words)))
    for word, row in zip(words, mat):
        print('{:>}\t{}'.format(word, " ".join('{:3}'.format(str(x)) for x in row)))