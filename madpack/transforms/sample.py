import numpy as np
from numpy.random import choice, normal


def split_indices(indices, seed, fraction, to_str=False):
    """ Splits the list of indices or data randomly (defined by seed) into two disjoint sets.
        to_str: pre-processing that converts data to strings.
    """

    if to_str:
        indices = [str(i) for i in indices]

    hashes = [hash(indices[i] + str(seed)) % 100000 for i in range(len(indices))]

    indices1 = [i for i in range(len(indices)) if hashes[i] < fraction * 100000]
    indices2 = [i for i in range(len(indices)) if hashes[i] >= fraction * 100000]

    return indices1, indices2


def sample_binned(sequence, bins, bin_randomness=0, fill_gaps=None):
    """

    Divides the input sequence into `k` bins and draws a random sample from each bin such that the resulting sequence
    has length `k`
    """

    assert fill_gaps in {None, 'prev'}

    eps = 10e-8

    if type(sequence) == int:
        sequence = np.arange(sequence)

    assert len(sequence) > 0

    # relative position of each element
    q = np.linspace(eps, 1, len(sequence))

    if bin_randomness > 0:
        q = np.clip(q + normal(0, bin_randomness / len(sequence)), eps, 1)

    bin_assignment = np.ceil(q*bins) - 1

    bins = [[] for _ in range(bins)]
    for s, b in zip(sequence, bin_assignment):
        bins[int(b)].append(s)

    indices = [choice(s) if len(s) > 0 else None for s in bins]

    if fill_gaps == 'prev':

        last_valid = None
        for i, j in enumerate(indices):
            if j is not None:
                last_valid = j
            else:
                indices[i] = last_valid

    return indices


def sample_equidistant(sequence, n_samples, offset=None):
    """
    Samples from sequence such that the interval between samples are equal.
    """

    if type(sequence) == int:
        sequence = np.arange(sequence)

    assert len(sequence) > 0

    # must be at least 1
    skip = max(1, int(np.floor(len(sequence) / n_samples)))

    offset_range = len(sequence) - skip * n_samples

    if offset is None:
        offset = choice(offset_range + 1)

    sampled_sequence = sequence[offset:offset + skip*n_samples:skip].tolist()

    # if more samples are needed, just add some Nones
    if len(sequence) < n_samples:
        sampled_sequence += [None] * (n_samples - len(sampled_sequence))
        log.warning('EQUIDISTANT SAMPLING: Too few samples for sequence')

    return sampled_sequence


def sample_spaced(sequence, n_samples, min_dist):
    """
    Sample from `sequence` such that indices have a distance of at least `min_dist`.

    The problem looks like this:
    let "|" be shapes and md the minimal distance: a | md + b1 | md + b2 | c
    now a, b1,.. and c can be chosen freely as long as: n_frames - n_shapes * md = a + b1 + ... + c

    We have some base_positions. These are the smallest indices possible for the respective sample.
    E.g. if the spacing is 3, base positions are: 0, 3, 6, ...
    To this we add randomly sampled offsets which need to sum to the remaining free space.

    Here the offsets are accumulative, i.e. each offset is larger than previous ones. By sampling from
    an arange and subsequent sorting, we obtain such offset that do not exceed the free space.

    """

    assert len(sequence) >= n_samples * min_dist

    free_space = np.arange(len(sequence) - (n_samples - 1) * (min_dist + 1))
    offsets = np.sort(choice(free_space, n_samples))
    base_positions = np.arange(n_samples) * (min_dist + 1)

    return offsets + base_positions


def sample_limited_repetitions(sequence, n_samples, max_repetitions):
    # trick to allow multiple assignments to a frame while considering the limit of self.multi_shape:
    # repeat the frame list and sample without replacement.
    return choice(list(range(len(sequence))) * max_repetitions, n_samples, replace=False)
