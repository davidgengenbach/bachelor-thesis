import sklearn
from transformers.phi_picker_transformer import PhiPickerTransformer
from transformers.fast_wl_graph_kernel_transformer import FastWLGraphKernelTransformer
from transformers.nx_graph_to_tuple_transformer import NxGraphToTupleTransformer

def get_pipeline():
    return sklearn.pipeline.Pipeline(
        [
            ('tuple_transformer', NxGraphToTupleTransformer()),
            ('fast_wl', FastWLGraphKernelTransformer(debug = False)),
            ('phi_picker', PhiPickerTransformer())
        ]
    )

def convert_graphs_to_tuples(X):
    trans = NxGraphToTupleTransformer()
    trans.transform(X)