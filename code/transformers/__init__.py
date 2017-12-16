from . import fast_wl_graph_kernel_transformer
from . import gram_matrix_transformer
from . import graph_multi_word_label_splitter
from . import graph_to_text_transformer
from . import nx_graph_to_tuple_transformer
from . import phi_picker_transformer
from . import preprocessing_transformer
from . import relabel_graphs_transformer
from . import remove_single_occurrence_graph_labels
from . import simple_preprocessing_transformer
from . import tuple_selector

from .fast_wl_graph_kernel_transformer import FastWLGraphKernelTransformer
from .gram_matrix_transformer import PhiListToGramMatrixTransformer
from .graph_multi_word_label_splitter import GraphMultiWordLabelSplitter
from .graph_to_text_transformer import GraphToTextTransformer
from .nx_graph_to_tuple_transformer import NxGraphToTupleTransformer
from .phi_picker_transformer import PhiPickerTransformer
from .preprocessing_transformer import SpacyPreProcessingTransformer
from .relabel_graphs_transformer import RelabelGraphsTransformer
from .remove_single_occurrence_graph_labels import RemoveInfrequentGraphLabels
from .remove_single_occurrence_graph_labels import RemoveInfrequentGraphLabels
from .simple_preprocessing_transformer import SimplePreProcessingTransformer
from .tuple_selector import TupleSelector

import sklearn
import sklearn.feature_extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Add all transformers to this module
TRANSFORMERS = {name: val for name, val in locals().items() if name[0].isupper()}