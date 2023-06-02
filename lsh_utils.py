#!/usr/bin/env python3

import pandas as pd
import numpy as np

from collections.abc import Hashable
from functools import lru_cache
from itertools import combinations
from itertools import chain
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from xxhash import xxh128
from tqdm.auto import tqdm

tqdm.monitor_interval = 1.0

DEFAULT_SEED = 0
DEFAULT_HASHSIZE = 256
DEFAULT_NGRAM_SIZE = 3
DEFAULT_WINDOW_SIZE = 2
CACHE_SIZE = 2 ** 20

def hashf(bytes_, bits=DEFAULT_HASHSIZE, seed=DEFAULT_SEED):
    """Fast underlying hashing function that can be sized to an arbitrary number of bits"""
    hash_bits = 0
    passes = 0
    xxh_size = 128 # number of bits in the underlying xxhash
    while (passes * xxh_size) < bits:
        hash_bits ^= xxh128(bytes_, seed=seed+passes).intdigest() << (xxh_size * passes)
        passes += 1
    # check if we have too many bits
    if hash_bits.bit_length() > bits:
        hash_bits >>= hash_bits.bit_length() - bits
    return hash_bits

class Hashable_Ndarray(Hashable, np.ndarray):
    """An np.ndarray subclass that is frozen (uneditable) and is hashable
    
    Cf. https://machineawakening.blogspot.com/2011/03/making-numpy-ndarrays-hashable.html"""
    
    hashsize = DEFAULT_HASHSIZE
    seed = DEFAULT_SEED
    
    def __new__(cls, values):
        return np.array(values, order='C').view(cls)
    
    def __init__(self, values):
        self.flags.writeable = False
    
    def __eq__(self, other):
        return np.all(np.ndarray.__eq__(self, other))
    
    def __hash__(self):
        return hashf(self, bits=self.hashsize, seed=self.seed)
    
    def __setitem__(self, key, value):
        raise Exception('hashable arrays are read-only')

def ngrams(iterable, n=2):
    """Generate ngrams from an iterable in a totally lazy fashion
    
    l = range(5)
    list(l) -> [0, 1, 2, 3, 4]
    list(ngrams(l, n=1)) -> [(0,), (1,), (2,), (3,), (4,)]
    list(ngrams(l, n=2)) -> [(0, 1), (1, 2), (2, 3), (3, 4)]
    list(ngrams(l, n=3)) -> [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    
    """
    from itertools import tee
    from itertools import islice
    yield from zip(*(
        islice(items, i, None) for i, items in
        enumerate(tee(iter(iterable), n))
    ))

@lru_cache(maxsize=CACHE_SIZE)
def segment_simhash(
    m,
    n=DEFAULT_NGRAM_SIZE,
    hashsize=DEFAULT_HASHSIZE,
    seed=DEFAULT_SEED
):
    """Compute a simhash over the bytes of n-grams of rows in a matrix
    
    This strategy uses a sliding window of size n that is iterated across the matrix
    
    E.g., for a 5 x 5 matrix, the 3-gram sliding windows are marked with "x"s below:
    
    iteration 0 | iteration 1 | iteration 2
    xxxxx       | .....       | .....      
    xxxxx       | xxxxx       | .....      
    xxxxx       | xxxxx       | xxxxx      
    .....       | xxxxx       | xxxxx      
    .....       | .....       | xxxxx      
    
    For a 2D matrix, one can perform a hash over the columns simply by passing in the transpose of the matrix (m.T)
    """
    lsh = np.zeros(hashsize)
    if len(m) < n: # too small
        return 0
    for ngram in ngrams(m, n=n):
        for j in range(hashsize):
            data = b''.join(segment.tobytes() for segment in ngram)
            if hashf(data, bits=hashsize, seed=seed) & (1 << j):
                lsh[j] += 1
            else:
                lsh[j] -= 1
    return sum(int(b > 0) << i for (i, b) in enumerate(reversed(lsh)))

@lru_cache(maxsize=CACHE_SIZE)
def stride_simhash(
    m,
    n=DEFAULT_NGRAM_SIZE,
    hashsize=DEFAULT_HASHSIZE,
    seed=DEFAULT_SEED
):
    """A simhash using a sliding window strategy for feature extraction.
    
    This strategy uses an n x n sliding window that is iterated across each axis of the matrix
    
    E.g., for a 5 x 5 matrix, the 3 x 3 sliding windows are marked with "x"s below:
    
    axis 0
    iteration 0 | iteration 1 | iteration 2
    xxx..       | .xxx.       | ..xxx      
    xxx..       | .xxx.       | ..xxx      
    xxx..       | .xxx.       | ..xxx      
    .....       | .....       | .....      
    .....       | .....       | .....      
    
    axis 1
    iteration 0 | iteration 1 | iteration 2
    .....       | .....       | .....
    xxx..       | .xxx.       | ..xxx
    xxx..       | .xxx.       | ..xxx
    xxx..       | .xxx.       | ..xxx
    .....       | .....       | .....
    
    axis 2
    iteration 0 | iteration 1 | iteration 2
    .....       | .....       | .....
    .....       | .....       | .....
    xxx..       | .xxx.       | ..xxx
    xxx..       | .xxx.       | ..xxx
    xxx..       | .xxx.       | ..xxx
    
    The underlying hashes of the bytes of each of the 3 x 3 views shown above are used to compute the simhash of the full matrix
    """
    lsh = np.zeros(hashsize)
    window_shape = (n, n)
    if m.shape < window_shape: # too small
        return 0
    for axis in sliding_window_view(m, window_shape=window_shape):
        for view in axis:
            for bit in range(hashsize):
                data = view.tobytes()
                if hashf(data, bits=hashsize, seed=seed) & (1 << bit):
                    lsh[bit] += 1
                else:
                    lsh[bit] -= 1
    return sum(int(bit > 0) << place for (place, bit) in enumerate(reversed(lsh)))

# simhash features
features = [
    stride_simhash,
    stride_simhash,
    segment_simhash,
    segment_simhash,
]
# matrix transforms
transforms = [
    lambda m: Hashable_Ndarray(m),   # for 2D stride-based n-gram features
    lambda m: Hashable_Ndarray(m.T), # for 2D stride-based n-gram features of the transpose
    lambda m: Hashable_Ndarray(m),   # for 2D column n-gram features
    lambda m: Hashable_Ndarray(m.T), # for 2D row n-gram features
]
assert len(features) == len(transforms)
# n-gram sizes
n_gram_sizes = [
    #1,
    #2,
    3,
    #4,
    5,
    #6,
    7,
    #8,
]

SIMHASH_BITS = DEFAULT_HASHSIZE * len(features) * len(n_gram_sizes) # actual simhash width in bits is dependent on hashsize, the number of features, and the number of n-gram sizes

@lru_cache(maxsize=CACHE_SIZE)
def pad(m, n):
    # we pad the "beginning" and "end" of the matrix with (n // 2) rows so that it will always
    # be large enough for our n-gram features to be informative
    left_padding = np.array([np.full(m[0].shape, fill_value='^')] * (n // 2))
    right_padding = np.array([np.full(m[0].shape, fill_value='$')] * (n // 2))
    padded_m = np.concatenate((left_padding, m, right_padding), axis=0)
    # we similarly pad the transpose "top" and "bottom" with (n // 2) columns
    top_padding = np.array([np.full(padded_m.T[0].shape, fill_value='&')] * (n // 2)).T
    bottom_padding = np.array([np.full(padded_m.T[0].shape, fill_value='%')] * (n // 2)).T
    padded_m = np.concatenate((top_padding, padded_m, bottom_padding), axis=1)
    return padded_m

@lru_cache(maxsize=CACHE_SIZE)
def matrix_simhash(
    m,
    hashsize=DEFAULT_HASHSIZE,
    seed=DEFAULT_SEED
):
    """Compute a simhash by XORing simhashes of the rows and columns of a phoneme matrix
    and also a simhash using a stride-based sliding window over the phoneme matrix"""
    simhash = 0
    # simhashes for different features are bit-shifted by multiples
    # of the bit width and the feature index so as not to interfere with one another
    #matrices = [transform(m) for transform in transforms]
    for i, (lsh, transform) in enumerate(zip(features, transforms)):
        for j in n_gram_sizes:
            simhash ^= lsh(transform(pad(m, j)), n=j, hashsize=hashsize, seed=seed) << hashsize * i * j # shift the bits left so bits from different features/n-gram sizes don't clobber one another
    return simhash

@lru_cache(maxsize=CACHE_SIZE)
def rotate(simhash, bits, rotations=1):
    """Bitwise rotate a simhash with bits bits"""
    rotations %= bits
    if rotations < 1:
        return simhash
    mask = (2 ** bits) - 1
    simhash &= mask
    return (simhash >> rotations) | (simhash << (bits - rotations) & mask)

def load_phoible(path='phoible.csv', cache=True):
    """Download the PHOIBLE data and load it as a dataframe suitable for further processing"""
    if not(cache) or not(Path(path).is_file()):
        phoible = pd.read_csv('https://raw.githubusercontent.com/phoible/dev/master/data/phoible.csv', low_memory=False)
        with open(path, mode='w') as f:
            print(phoible.to_csv(), file=f)
    return pd.read_csv(path, low_memory=False)

phoible = load_phoible()

def simdiff(a, b):
    """Compute the bitwise difference between two simhashes"""
    bits = max(a.bit_length(), b.bit_length())
    if bits < 1:
        raise ValueError(f'need at least 1 bit to compute bitwise simhash difference (bits={bits})')
    xor = a ^ b
    difference = sum(((xor & (1 << i)) > 0) for i in range(bits))
    return difference

def candidates(
    tokens,
    simhash_bits=SIMHASH_BITS,
    window=DEFAULT_WINDOW_SIZE
):
    """Generate candidate token pairs and their LSH bitwise differences
    
    Will generate tuples of the form: ((a:Token, b:Token), difference:int)
    """
    d = set()
    # rotate over each bit in the simhash
    for i in range(simhash_bits):
        def lsh(token):
            return rotate(token.simhash(), simhash_bits, rotations=i)
        for ngram in ngrams(sorted(tokens, key=lsh), n=window):
            # check each pairwise combination within the window
            for a, b in combinations(ngram, 2):
                key = tuple(sorted((a, b)))
                if key not in d:
                    token_a, token_b = key
                    difference = simdiff(lsh(a), lsh(b))
                    yield key, difference
                    d.add(key)

def candidates(
    queries,
    tokens,
    simhash_bits=SIMHASH_BITS,
    window=DEFAULT_WINDOW_SIZE
):
    """Generate candidate token pairs and their LSH bitwise differences
    
    Will generate tuples of the form: ((a:Token, b:Token), difference:int)
    """
    d = set()
    # rotate over each bit in the simhash
    for i in range(simhash_bits):
        def lsh(token):
            return rotate(token.simhash(), simhash_bits, rotations=i)
        for ngram in ngrams(sorted(chain(queries, tokens), key=lsh), n=window):
            if not(any(queries)) or (set(ngram) & set(queries)):
                # check each pairwise combination within the window
                for a, b in combinations(ngram, 2):
                    if not(any(queries)) or ({a, b} & set(queries)):
                        key = tuple(sorted((a, b)))
                        if key not in d:
                            token_a, token_b = key
                            difference = simdiff(lsh(a), lsh(b))
                            yield key, difference
                            d.add(key)

def compare(
    tokens,
    simhash_bits=SIMHASH_BITS,
    window=DEFAULT_WINDOW_SIZE
):
    """Generate comparison records considering all tokens
    
    Records contain the following:
    "a": the first token of the pair
    "b": the second token of the pair
    "simhash difference (in bits)": the bitwise difference between the simhash of a and the simhash of b
    "sigma(phonemic)": the phonemic similarity score for the pair (a, b) computed by: 1 - (difference/(2*bits)
    """
    queries = set()
    with tqdm(
        candidates(
            queries,
            tokens,
            simhash_bits=simhash_bits,
            window=window
        ),
        desc='comparing pairs',
        unit='pair',
        miniters=1,
        mininterval=0,
        dynamic_ncols=True,
        disable=None,
    ) as pairs:
        for ((a, b), difference) in pairs:
            yield {
                'a': a,
                'b': b,
                'simhash difference (in bits)': difference,
                'similarity score': f'{1.0 - (difference/simhash_bits):0.3}'
            }

def search(
    queries,
    tokens,
    simhash_bits=SIMHASH_BITS,
    window=DEFAULT_WINDOW_SIZE
):
    """Generate comparison records considering only queries against all tokens
    
    Records contain the following:
    "a": the first token of the pair
    "b": the second token of the pair
    "simhash difference (in bits)": the bitwise difference between the simhash of a and the simhash of b
    "sigma(phonemic)": the phonemic similarity score for the pair (a, b) computed by: 1 - (difference/(2*bits)
    """
    with tqdm(
        candidates(
            queries,
            tokens,
            simhash_bits=simhash_bits,
            window=window
        ),
        desc='querying',
        unit='pair',
        miniters=1,
        mininterval=0,
        dynamic_ncols=True,
        disable=None,
    ) as pairs:
        for ((a, b), difference) in pairs:
            yield {
                'a': a,
                'b': b,
                'simhash difference (in bits)': difference,
                'similarity score': f'{1.0 - (difference/simhash_bits):0.3}'
            }
