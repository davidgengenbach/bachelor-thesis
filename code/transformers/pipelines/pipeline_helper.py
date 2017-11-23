def add_prefix_to_params(params: dict, prefix: str) -> dict:
    return {
        prefix + k: v for k, v in params.items()
    }


def flatten_nested_params(params: dict):
    if not isinstance(params, dict): return params

    out = dict()
    for k, v in params.items():
        if isinstance(v, list):
            out[k] = v
        elif isinstance(v, dict):
            for k_, v_ in v.items():
                key = k + '__' + k_
                flatted = flatten_nested_params(v_)
                if isinstance(flatted, list):
                    out[key] = flatted
                elif isinstance(flatted, dict):
                    for k__, v__ in flatted.items():
                        key = key + '__' + k__
                        out[key] = v__
                else:
                    raise Exception('Invalid params type: "{}" (type={})'.format(flatted, type(flatted)))
    return out


def remove_complex_types(params):
    out = {}
    for k, v in params.items():
        assert isinstance(v, list)

        def is_complex_type(x):
            return not isinstance(x, (int, float, str, bool, tuple))

        v = [v_ if not is_complex_type(v_) else type(v_).__name__ for v_ in v]
        out[k] = v
    return out


def remove_complex_types_simple(params):
    out = {}
    for k, v in params.items():
        def is_complex_type(x):
            return not isinstance(x, (int, float, str, bool, tuple))

        v = v if not is_complex_type(v) else type(v).__name__
        out[k] = v
    return out
