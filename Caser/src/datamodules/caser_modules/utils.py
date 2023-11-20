import numpy as np

def shuffle(*arrays, **kwargs):
    """
    Shuffle input with the same length.
    Args:
        *arrays ():
        **kwargs (): you can specify 'indices' parameter to add it to output.

    Returns: shuffled arrays, shuffled indices (if 'indices' is True)

    """
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result