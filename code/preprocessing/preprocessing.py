from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords

import re

ALPHA_NUM_REGEX = re.compile(r'[^a-zA-Z0-9 \.\?!,\'\-]')

LINEBREAK_REGEX = re.compile(r'((\r\n)|[\n\v])+')
NONBREAKING_SPACE_REGEX = re.compile(r'(?!\n)\s+')
NUM_REGEXP = re.compile(r'\d+')

control_chars = '\x00-\x1f\x7f-\x9f'
control_char_re = re.compile('[%s]' % re.escape(control_chars))

nlp = None


def init_spacy():
    import spacy
    global nlp
    if nlp: return
    nlp = spacy.load('en')


def get_spacy_parse(texts, batch_size=100, n_threads=1):
    init_spacy()
    return nlp.pipe(texts, batch_size=batch_size, n_threads=n_threads)


def preprocess(t):
    fns = [
        remove_non_alphanum,
        remove_non_printable,
        number_to_placeholder,
        ana_preprocessing
    ]

    for fn in fns:
        t = fn(t)
    return t


def ana_preprocessing(text, lower_case=False, keep_newlines=False):
    # all lowercase
    if lower_case:
        text = text.lower()

    # replace TAB, NEWLINE and RETURN characters by SPACE
    replacements = (
        ('\t', ' '),
        ('\n', ' '),
        ('\r', ' '),
    )

    for d in replacements:
        if keep_newlines and d[0] == '\n': continue
        text = text.replace(*d)

    text = normalize_whitespace(text)
    return text


def preprocess_text_spacy(
        texts,
        n_jobs=1,
        batch_size=100
):
    """Preprocesses text by first using the default preprocess function (removing newlines etc.) then parse using spacy.

    Args:
        texts (list of str): list of texts
        n_jobs (int, optional): how many threads should be used
        batch_size (int, optional): how many texts should be done per thread

    Returns:
        list of str: the pre-processed text
    """
    texts = [preprocess(t) for t in texts]
    return list(get_spacy_parse(texts, batch_size=batch_size, n_threads=n_jobs))


def to_tokens(text):
    return word_tokenize(text.lower())


def normalize_whitespace(text):
    """
    Given ``text`` str, replace one or more spacings with a single space, and one
    or more linebreaks with a single newline. Also strip leading/trailing whitespace.
    Taken from http://textacy.readthedocs.io/en/latest/_modules/textacy/preprocess.html
    """
    return NONBREAKING_SPACE_REGEX.sub(' ', LINEBREAK_REGEX.sub(r'\n', text)).strip()


def remove_stopwords(tokens, stopwords=None):
    if not stopwords:
        stopwords = set(nltk_stopwords.words('english'))

    return [i for i in tokens if i not in stopwords]


def remove_non_printable(text):
    return control_char_re.sub('', text)


def number_to_placeholder(text):
    return NUM_REGEXP.sub(' NUMBER ', text)


def remove_non_alphanum(text):
    return ALPHA_NUM_REGEX.sub(' ', text)
