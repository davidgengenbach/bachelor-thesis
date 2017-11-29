import gensim
import numpy as np


def merge_lookups(*lookups):
    """Merges two dicts. Order is important: first one can not be overwritten (?)
    
    Args:
        *lookups: dicts
    
    Returns:
        dict: the merged lookup
    """
    root_lookup = {}
    for lookup in reversed(lookups):
        root_lookup.update(lookup)
    return root_lookup


def get_embedding_model(w2v_file, binary = False, first_line_header = True, with_gensim = False, datatype=np.float64):
    import gensim
    if binary or with_gensim:
        embeddings = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=binary)
    else:
        with open(w2v_file) as f:
            if first_line_header:
                first_line = f.readline()
                num_labels, num_dim = [int(x) for x in first_line.split(' ')]
            embeddings = {x.split(' ', 1)[0].strip(): x.split(' ', 1)[1].strip() for x in f}
    return embeddings

def get_embeddings_for_labels(labels, embedding, check_most_similar = False, restrict_vocab = None, lookup_embedding = None, topn = 20, solve_composite_labels = True):
    """Returns the embeddings for given labels. Optionally also tries to resolve missing embeddings with another embedding, the lookup_embedding, by looking for the most_similar words that are also in the original embedding.
    
    Args:
        labels (list(str)): A list of strings
        embedding (gensim.models.keyedvectors.KeyedVectors): the embedding to search the labels in
        check_most_similar (bool, optional): Whether to try to try to find a similar word in another embedding, the lookup_embedding
        restrict_vocab (list(str), optional): The labels that can be looked up in the lookup_embedding - these are most likely the labels that are in both the embedding and lookup_embedding
        lookup_embedding (gensim.models.keyedvectors.KeyedVectors, optional): The backup embedding to search similar words that are also in the original embedding
        topn (int, optional): How many most similar words to retrieve from the lookup embedding
    
    Returns:
        tuple(dict, list(str)): the dictionary with the labels as keys and the embeddings for that labels as value. And a list of label that could not be found
    """
    assert not check_most_similar or lookup_embedding is not None
    not_found, embeddings, lookup, similar_els = [], {}, {}, {}

    embedding_size = lookup_embedding.syn0.shape[1]
    for label in labels:
        label = str(label).lower()
        is_composite = label.count(' ') > 0
        if label in embedding:
            embeddings[label] = embedding[label]
            lookup[label] = label
        elif check_most_similar and (label in lookup_embedding or (solve_composite_labels and is_composite)):
            if solve_composite_labels and label.count(' ') > 0:
                label_parts = label.split(' ')
                composite_vector = np.zeros(embedding_size, dtype=np.float128)
                for label_part in label_parts:
                    if label_part in lookup_embedding:
                        embedding_ = lookup_embedding[label_part]
                        composite_vector += embedding_
                if len(composite_vector.nonzero()[0]):
                    most_similar = lookup_embedding.similar_by_vector(composite_vector, topn = topn)
                else:
                    most_similar = []
            else:
                most_similar = lookup_embedding.similar_by_word(label, topn = topn)

            if len(most_similar):
                similar_els[label] = most_similar

            most_similar_labels = [label for label, similarity in most_similar]
            match = set(most_similar_labels) & set(restrict_vocab)
            if len(match):
                most_similar_label_found = [label for label, similarity in most_similar if label in match][0]
                embeddings[label] = embedding[most_similar_label_found]
                lookup[label] = most_similar_label_found
            else:
                not_found.append(label)
                lookup[label] = label
        else:
            not_found.append(label)
    return embeddings, not_found, lookup, similar_els


def get_embeddings_for_labels_with_lookup(all_labels, trained_embedding, pre_trained_embedding, solve_composite_labels = True):

    all_labels = set(all_labels)
    if solve_composite_labels:
        composite_labels = set([str(label) for label in all_labels if str(label).strip().count(' ') > 0])
        for label in composite_labels:
            all_labels |= set(label.split(' '))

    embeddings_trained_labels = set(trained_embedding.vocab.keys())
    not_found_trained = set(all_labels) - embeddings_trained_labels

    embeddings_pre_trained_labels = set(pre_trained_embedding.vocab.keys())
    not_found_pre_trained = set(all_labels) - embeddings_pre_trained_labels

    in_both = embeddings_trained_labels & embeddings_pre_trained_labels

    embeddings_pre_trained, not_found_pre_trained_coreferenced, lookup, similar_els = get_embeddings_for_labels(all_labels, pre_trained_embedding, check_most_similar = True, restrict_vocab = in_both, lookup_embedding = trained_embedding)

    return embeddings_pre_trained, not_found_pre_trained_coreferenced, not_found_trained, not_found_pre_trained, lookup, similar_els


def save_embedding_dict(embedding, filename):
    num_embeddings = len(embedding.keys())
    num_dim = len(embedding[list(embedding.keys())[0]])
    with open(filename, 'w') as f:
        f.write('{} {}\n'.format(num_embeddings, num_dim))
        f.write("\n".join(['"{}" {}'.format(label, ' '.join([str(x) for x in vec])) for label, vec in embedding.items() if label.strip() != '']))


# Taken and adapted from: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/keyedvectors.py
from gensim import utils, matutils 
from gensim.corpora.dictionary import Dictionary
from six import string_types, iteritems
from six.moves import xrange
from scipy import stats
from numpy import zeros, ascontiguousarray

REAL = np.float32

from utils.logger import LOGGER as logger
from gensim.models.word2vec import Vocab

def load_word2vec_format(cls = gensim.models.KeyedVectors, fname='', fvocab=None, binary=False, encoding='utf8', unicode_errors='strict',
                             limit=None, datatype=REAL):
        """


        Load the input-hidden weight matrix from the original C word2vec-tool format.
        Note that the information stored in the file is incomplete (the binary tree is missing),
        so while you can query for word similarity etc., you cannot continue training
        with a model loaded this way.
        `binary` is a boolean indicating whether the data is in binary word2vec format.
        `norm_only` is a boolean indicating whether to only store normalised word2vec vectors in memory.
        Word counts are read from `fvocab` filename, if set (this is the file generated
        by `-save-vocab` flag of the original C tool).
        If you trained the C model using non-utf8 encoding for words, specify that
        encoding in `encoding`.
        `unicode_errors`, default 'strict', is a string suitable to be passed as the `errors`
        argument to the unicode() (Python 2.x) or str() (Python 3.x) function. If your source
        file may include word tokens truncated in the middle of a multibyte unicode character
        (as is common from the original word2vec.c tool), 'ignore' or 'replace' may help.
        `limit` sets a maximum number of word-vectors to read from the file. The default,
        None, means read all.
        `datatype` (experimental) can coerce dimensions to a non-default float type (such
        as np.float16) to save memory. (Such types may result in much slower bulk operations
        or incompatibility with optimized routines.)
        """
        counts = None
        if fvocab is not None:
            logger.info("loading word counts from %s", fvocab)
            counts = {}
            with utils.smart_open(fvocab) as fin:
                for line in fin:
                    word, count = utils.to_unicode(line).strip().split()
                    counts[word] = int(count)

        logger.info("loading projection weights from %s", fname)
        with utils.smart_open(fname) as fin:
            header = utils.to_unicode(fin.readline(), encoding=encoding)
            vocab_size, vector_size = map(int, header.split())  # throws for invalid file format
            if limit:
                vocab_size = min(vocab_size, limit)
            result = cls()
            result.vector_size = vector_size
            result.syn0 = zeros((vocab_size, vector_size), dtype=datatype)

            def add_word(word, weights):
                word_id = len(result.vocab)
                if word in result.vocab:
                    logger.warning("duplicate word '%s' in %s, ignoring all but first", word, fname)
                    return
                if counts is None:
                    # most common scenario: no vocab file given. just make up some bogus counts, in descending order
                    result.vocab[word] = Vocab(index=word_id, count=vocab_size - word_id)
                elif word in counts:
                    # use count from the vocab file
                    result.vocab[word] = Vocab(index=word_id, count=counts[word])
                else:
                    # vocab file given, but word is missing -- set count to None (TODO: or raise?)
                    logger.warning("vocabulary file is incomplete: '%s' is missing", word)
                    result.vocab[word] = Vocab(index=word_id, count=None)
                result.syn0[word_id] = weights
                result.index2word.append(word)

            if binary:
                # TODO: delegate
                pass
            else:
                for line_no in xrange(vocab_size):
                    line = fin.readline()
                    if line == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if '"' in utils.to_unicode(line, encoding=encoding, errors=unicode_errors):
                        line = utils.to_unicode(line, encoding=encoding, errors=unicode_errors)
                        label = line.split('"', 1)[1].rsplit('"')[0].strip()
                        other = line.rsplit('"', 1)[1].strip().split(' ')
                        parts = [label] + other
                    else:
                        parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")
                    if len(parts) != vector_size + 1:
                        raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))

                    word, weights = parts[0], list(map(REAL, parts[1:]))
                    add_word(word, weights)

        if result.syn0.shape[0] != len(result.vocab):
            logger.info(
                "duplicate words detected, shrinking matrix size from %i to %i",
                result.syn0.shape[0], len(result.vocab)
            )
            result.syn0 = ascontiguousarray(result.syn0[: len(result.vocab)])
        assert (len(result.vocab), vector_size) == result.syn0.shape

        logger.info("loaded %s matrix from %s" % (result.syn0.shape, fname))
        return result
