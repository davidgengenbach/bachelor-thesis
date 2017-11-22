from utils import helper, primes
import numpy as np
import scipy.sparse


def transform(
        adjs,
        labels,
        h=2,
        rounding_factor=10,
        labels_dtype=np.uint64,
        node_weight_factors=None,
        use_early_stopping=True
):
    labels_flat = helper.flatten_array(labels)
    all_nodes = set(labels_flat)
    label_2_id = {label: idx for idx, label in enumerate(sorted(all_nodes, key=lambda x: str(x)))}
    id_2_label = {idx: label for label, idx in label_2_id.items()}
    labels = [[label_2_id[x] for x in xs] for xs in labels]
    labels_ = np.copy(labels)

    assert len(labels_) == len(adjs)
    assert not node_weight_factors or len(node_weight_factors) == len(adjs)

    num_nodes = sum([len(label) for label in labels])
    label_indices = [adj.shape[0] for adj in adjs]
    acc = 0
    for idx, num_els in enumerate(label_indices):
        label_indices[idx] = acc
        acc += num_els
    label_indices.append(acc)

    log_primes = primes.get_log_primes()

    if node_weight_factors:
        for x in node_weight_factors:
            assert np.all([isinstance(y, (int, float)) for y in x])

    if rounding_factor == -1:
        _rounding_factor = 1
    else:
        _rounding_factor = np.power(10, rounding_factor)

    def phi_list_to_sparse(phis):
        rows = []
        cols = []
        data = []
        for graph_idx, labels in enumerate(phis):
            rows += [graph_idx] * len(labels)
            cols += list(labels)
            if node_weight_factors:
                factor = node_weight_factors[graph_idx]
                data_ = factor
            else:
                data_ = [1] * len(labels)
            data += list(data_)
        phi = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(len(phis), num_nodes), dtype=np.uint16).tocsr()
        return phi

    def get_signatures(adj, labels):
        return ((labels + adj * log_primes[labels]) * _rounding_factor).astype(labels_dtype)

    highest_labels = [-1]

    def get_new_labels(labels_):
        new_labels = [get_signatures(adj, label) for adj, label in zip(adjs, labels_)]
        new_labels_flat = np.hstack(new_labels)
        unique, unique_indices, unique_inverse, unique_counts = np.unique(new_labels_flat, return_index=True, return_inverse=True, return_counts=True)
        highest_label = len(unique)
        highest_labels.append(highest_label)
        labels_ = [unique_inverse[label_indices[idx]:label_indices[idx + 1]] for idx in range(len(labels_))]
        return labels_

    phis = [labels_]
    for iteration in range(h):
        labels = phis[-1]
        new_labels = get_new_labels(labels)
        last_highest_label, current_highest_label = highest_labels[-2:]
        # Convergence
        if use_early_stopping and last_highest_label == current_highest_label:
            break
        phis.append(new_labels)
    return [phi_list_to_sparse(phi) for phi in phis]
