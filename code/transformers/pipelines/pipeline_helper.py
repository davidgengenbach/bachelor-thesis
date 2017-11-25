import numpy as np

def add_prefix_to_params(params: dict, prefix: str) -> dict:
    return {
        prefix + k: v for k, v in params.items()
    }


def flatten_nested_params(params: dict):
    if not isinstance(params, dict): return params

    # Already flattened
    if np.all([isinstance(x, list) or x is None for x in params.values()]):
        return params

    out = dict()
    for k, v in params.items():
        if isinstance(v, list):
            out[k] = v
        elif isinstance(v, dict):
            for k_, v_ in v.items():
                key = k + '__' + k_
                flatted = flatten_nested_params(v_)
                if isinstance(flatted, list) or flatted is None:
                    out[key] = flatted
                elif isinstance(flatted, dict):
                    for k__, v__ in flatted.items():
                        key_ = key + '__' + k__
                        out[key_] = v__
                else:
                    raise Exception('Invalid params type: "{}" (type={})'.format(flatted, type(flatted)))
    return out

def unflatten_params(params: dict)->dict:
    out = {}
    for param_key, val in params.items():
        assert val is None or isinstance(val, list)
        parts = param_key.split('__')
        current = out
        for idx, part in enumerate(parts):
            if part in current:
                current = current[part]
                continue
            else:
                current[part] = {}
            if idx == len(parts) - 1:
                current[part] = val
                continue
            current = current[part]
    return out


def is_complex_type(x):
    return not isinstance(x, (int, float, str, bool, tuple))

def remove_complex_types(params):
    out = {}
    for k, v in params.items():
        if v is None:
            out[k] = v
            continue
        assert isinstance(v, list)

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
