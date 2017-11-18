import sklearn
from transformers.pipelines import text_pipeline
from transformers.tuple_selector import TupleSelector
from transformers.phi_picker_transformer import PhiPickerTransformer
from transformers.fast_wl_graph_kernel_transformer import FastWLGraphKernelTransformer

from . import pipeline_helper


def get_params(reduced = False):
    fast_wl_estimator, fast_wl_params = get_fast_wl_params(reduced = reduced)

    pipeline = sklearn.pipeline.Pipeline([
        ('feature_extraction', fast_wl_estimator),
        ('normalizer', None),
        ('classifier', None)
    ])

    params = pipeline_helper.flatten_nested_params(dict(
        feature_extraction=fast_wl_params,
        normalizer=[sklearn.preprocessing.MaxAbsScaler()]
    ))

    return pipeline, params


def get_combined_params(text_reduced=True, graph_reduced=True):
    graph_estimator, graph_params = get_params(reduced=graph_reduced)
    text_estimator, text_params = text_pipeline.get_params(reduced=text_reduced)

    # Params
    params = pipeline_helper.flatten_nested_params(dict(
        features__fast_wl_pipeline__feature_extraction=graph_params,
        features__text__vectorizer=text_params
    ))

    # Pipeline
    combined_features = sklearn.pipeline.FeatureUnion([
        ('text', sklearn.pipeline.Pipeline([
            ('selector', TupleSelector(tuple_index=1)),
            ('vectorizer', text_estimator),
        ])),
        ('fast_wl_pipeline', sklearn.pipeline.Pipeline([
            ('selector', TupleSelector(tuple_index=0, v_stack=False)),
            ('feature_extraction', graph_estimator),
        ]))
    ])

    pipeline = sklearn.pipeline.Pipeline([
        ('features', combined_features),
        ('classifier', None)
    ])

    return pipeline, params


def get_fast_wl_params(reduced = False):
    pipeline = sklearn.pipeline.Pipeline([
        ('fast_wl', FastWLGraphKernelTransformer()),
        ('phi_picker', PhiPickerTransformer()),
    ])

    params = dict(
        fast_wl__h=[5],
        fast_wl__phi_dim=[None],
        fast_wl__round_to_decimals=[-1, 10],
        phi_picker__return_iteration=['stacked']
    )

    if reduced:
        params['fast_wl__round_to_decimals'] = [10]

    return pipeline, params


def add_num_vertices_to_fast_wl_params(params, num_vertices):
    for key, val in params.items():
        if key.endswith('phi_dim'):
            params[key] = [num_vertices]
            return

    raise Exception('Could not set phi_dim, key not in params: {}'.format(params))
