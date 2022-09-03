#!/usr/bin/env python3

import pandas as pd
import numpy as np

from collections.abc import Hashable
from functools import lru_cache
from itertools import combinations
from numpy.lib.stride_tricks import sliding_window_view
from operator import itemgetter
from pathlib import Path
from xxhash import xxh128

import sys

DEFAULT_SALT = 0
DEFAULT_HASHSIZE = 256
DEFAULT_NGRAM_SIZE = 4
DEFAULT_WINDOW_SIZE = 2

def hashf(bytes_, bits=DEFAULT_HASHSIZE, salt=DEFAULT_SALT):
    """Fast underlying hashing function that can be sized to an arbitrary number of bits"""
    hash_bits = 0
    passes = 0
    xxh_size = 128 # number of bits in the underlying xxhash
    while (passes * xxh_size) < bits:
        hash_bits ^= xxh128(bytes_, seed=salt+passes).intdigest() << (xxh_size * passes)
        passes += 1
    # check if we have too many bits
    if hash_bits.bit_length() > bits:
        hash_bits >>= hash_bits.bit_length() - bits
    return hash_bits

class Hashable_Ndarray(Hashable, np.ndarray):
    """An np.ndarray subclass that is frozen (uneditable) and is hashable
    
    Cf. https://machineawakening.blogspot.com/2011/03/making-numpy-ndarrays-hashable.html"""
    
    hashsize = DEFAULT_HASHSIZE
    salt = DEFAULT_SALT
    
    def __new__(cls, values):
        return np.array(values, order='C').view(cls)
    
    def __init__(self, values):
        self.flags.writeable = False
    
    def __eq__(self, other):
        return np.all(np.ndarray.__eq__(self, other))
    
    def __hash__(self):
        return hashf(self, bits=self.hashsize, salt=self.salt)
    
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

@lru_cache
def segment_simhash(
    m,
    n=DEFAULT_NGRAM_SIZE,
    hashsize=DEFAULT_HASHSIZE,
    salt=DEFAULT_SALT
):
    """Compute a simhash over the bytes of n-grams of rows in a matrix
    
    This strategy uses a 1D sliding window of size n that is iterated across the matrix
    
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
            if hashf(data, bits=hashsize, salt=salt) & (1 << j):
                lsh[j] += 1
            else:
                lsh[j] -= 1
    return sum(int(b > 0) << i for (i, b) in enumerate(reversed(lsh)))

@lru_cache
def stride_simhash(
    m,
    n=DEFAULT_NGRAM_SIZE,
    hashsize=DEFAULT_HASHSIZE,
    salt=DEFAULT_SALT
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
    for i, axis in enumerate(sliding_window_view(m, window_shape=window_shape)):
        for view in axis:
            for j in range(hashsize):
                data = view.tobytes()
                if hashf(data, bits=hashsize, salt=salt) & (1 << j):
                    lsh[j] += 1
                else:
                    lsh[j] -= 1
    return sum(int(b > 0) << i for (i, b) in enumerate(reversed(lsh)))

lsh_features = [
    stride_simhash,
    segment_simhash,
    segment_simhash,
]

matrix_transforms = [
    lambda m: Hashable_Ndarray(m),   # for 2D stride-based n-gram features
    #lambda m: Hashable_Ndarray(m.T), # for 2D row n-gram features
    #lambda m: Hashable_Ndarray(m),   # for 2D column n-gram features
]

@lru_cache
def matrix_simhash(
    m,
    n=DEFAULT_NGRAM_SIZE,
    hashsize=DEFAULT_HASHSIZE,
    salt=DEFAULT_SALT
):
    """Compute a simhash by XORing simhashes of the rows and columns of a phoneme matrix
    and also a simhash using a stride-based sliding window over the phoneme matrix"""
    simhash = 0
    # we pad the beginning and end of the matrix (n // 2) rows so that it will always
    # be large enough for our n-gram features to be informative
    left_padding = [np.full(m[0].shape, fill_value='^')] * (n // 2) 
    right_padding = [np.full(m[0].shape, fill_value='$')] * (n // 2) 
    m = np.concatenate((left_padding, m, right_padding), axis=0)
    # iterate over variable n-gram sizes from [2...n] (we skip 1-grams because they aren't informative enough)
    for i in range(2, n+1):
        # simhashes for different features are bit-shifted by multiples
        # of the bit width and the feature index so as not to interfere with one another
        matrices = [
            transform(m)
            for transform in matrix_transforms
        ]
        for j, (lsh, matrix) in enumerate(zip(lsh_features, matrices)):
            simhash ^= lsh(matrix, n=i, hashsize=hashsize, salt=salt) << i * j * hashsize # shift the bits left so bits from different features/n-gram sizes don't clobber one another
    return simhash

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

def ranked_pairs(
    tokens,
    n=DEFAULT_NGRAM_SIZE,
    hashsize=DEFAULT_HASHSIZE,
    window=DEFAULT_WINDOW_SIZE
):
    """Generate ranked pairs of tokens sorted by their LSH similarity
    
    Will generate records of the form: ((a:Token, b:Token), difference:int)
    """
    d = {}
    actual_bitwidth = hashsize * (n - 1) * len(lsh_features) # actual simhash width in bits is dependent on hashsize, the range over n, and the number of features
    # rotate over each bit in the simhash
    for i in range(actual_bitwidth):
        def lsh(token):
            return token.simhash_rotate(rotations=i, n=n)
        for ngram in ngrams(sorted(tokens, key=lsh), n=window):
            # check each pairwise combination within the window
            for a, b in combinations(ngram, 2):
                key = tuple(sorted((a, b)))
                if key not in d:
                    token_a, token_b = key
                    d[key] = simdiff(lsh(a), lsh(b))
    yield from sorted(d.items(), key=itemgetter(1, 0))

def compare(
    tokens,
    n=DEFAULT_NGRAM_SIZE,
    hashsize=DEFAULT_HASHSIZE,
    window=DEFAULT_WINDOW_SIZE
):
    """Return a datafram comparing pairs of Tokens sorted by phonemic similarity
    
    Dataframe columns:
    "a": the first token of the pair
    "b": the second token of the pair
    "simhash difference (in bits)": the bitwise difference between the simhash of a and the simhash of b
    "sigma(phonemic)": the phonemic similarity score for the pair (a, b) computed by: 1 - (difference/(2*bits)
    """
    try:
        get_ipython
        from tqdm.notebook import tqdm
    except:
        from tqdm import tqdm
    pairs = tqdm(
        list(ranked_pairs(tokens, n=n, hashsize=hashsize, window=window)),
        desc='ranking pairs by bitwise similarity',
        unit='pair'
    )
    actual_bitwidth = hashsize * (n - 1) * len(lsh_features) # actual simhash width in bits is dependent on hashsize, the range over n, and the number of features
    return pd.DataFrame(
        {
            'a': a,
            'b': b,
            'simhash difference (in bits)': difference,
            'similarity score': f'{1.0 - (difference/actual_bitwidth):0.3}'
        } for ((a, b), difference) in pairs
    )
