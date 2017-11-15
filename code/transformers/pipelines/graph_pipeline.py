from utils import graph_helper
from transformers import fast_wl_pipeline, text_pipeline
from transformers.tuple_selector import TupleSelector
import sklearn
import networkx as nx

def get_combined_pipeline():

    graph_fast_wl_grid_params = {
        'fast_wl__h': [],
        'fast_wl__phi_dim': [],
        'fast_wl__round_to_decimals': [],
        'fast_wl__use_node_weight_factors': [True, False],
        'phi_picker__return_iteration': [],
        'normalizer': [],
    }

    grid_params_combined = {
        'features__fast_wl_pipeline__feature_extraction__' + k: val for k, val in graph_fast_wl_grid_params.items()
    }

    grid_params_combined = dict(
        grid_params_combined,
        **{
            'features__text__vectorizer__' + k: val
            for k, val in text_pipeline.get_param_grid(reduced=True).items()
        }
    )

    combined_features = sklearn.pipeline.FeatureUnion([
        ('text', sklearn.pipeline.Pipeline([
            ('selector', TupleSelector(tuple_index=1)),
            ('vectorizer', text_pipeline.get_pipeline()),
        ])),
        ('fast_wl_pipeline', sklearn.pipeline.Pipeline([
            ('selector', TupleSelector(tuple_index=0, v_stack=False)),
            ('feature_extraction', fast_wl_pipeline.get_pipeline()),
        ]))
    ])

    pipeline = sklearn.pipeline.Pipeline([
        ('features', combined_features),
        # ('normalizer', None),
        ('classifier', None)
    ])

    return pipeline, grid_params_combined