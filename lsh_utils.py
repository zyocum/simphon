#!/usr/bin/env python3

import pandas as pd
import numpy as np
import mmh3

from itertools import combinations
from numpy.lib.stride_tricks import sliding_window_view
from operator import itemgetter
from pathlib import Path

hashes = {
    32: mmh3.hash, # 32 bit murmurhash
    64: lambda x: mmh3.hash64(x)[0], # mmh3.hash64 returns two 64 bit hashes, so we take the first one
    128: mmh3.hash128 # 128 bit murmurhash
}

n = 3
hashsize = 128
window = 10

def ngrams(iterable, n=n):
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

def segment_simhash(m, n=n, bits=hashsize):
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
    hashf = hashes[bits]
    lsh = np.zeros(bits)
    n = min(len(m), n)
    for ngram in ngrams(m, n=n):
        for j in range(bits):
            data = b'\x00' + b''.join(segment.tobytes() for segment in ngram)
            if hashf(data) & (1 << j):
                lsh[j] += 1
            else:
                lsh[j] -= 1
    return sum(int(b > 0) << i for (i, b) in enumerate(reversed(lsh)))

def stride_simhash(m, n=n, bits=hashsize):
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
    hashf = hashes[bits]
    lsh = np.zeros(bits)
    window_shape = (n, n)
    if m.shape < window_shape: # too small
        return 0
    for i, axis in enumerate(sliding_window_view(m, window_shape=window_shape)):
        for view in axis:
            for j in range(bits):
                data = int.to_bytes(i, length=1, byteorder='big') + view.tobytes()
                if hashf(data) & (1 << j):
                    lsh[j] += 1
                else:
                    lsh[j] -= 1
    return sum(int(b > 0) << i for (i, b) in enumerate(reversed(lsh)))

def matrix_simhash(m, n=n, bits=hashsize):
    """Compute a simhash by XORing simhashes of the rows and columns of a phoneme matrix
    and also a simhash based on a stride-based sliding window over the phoneme matrix"""
    # the lower bits are a simhash(rows) XOR simhash(columns)
    simhash = segment_simhash(m, n=n, bits=bits) # simhash of rows
    simhash ^= segment_simhash(m.T, n=n, bits=bits) # simhash of columns
    # the stride simhash bits are bitshifted left so as not to overl the row/column bits
    simhash ^= stride_simhash(m, n=n, bits=bits) << bits
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

def ranked_pairs(tokens, n=n, bits=hashsize, window=window):
    """Rank pairs of phonemic sequences sorted by LSH similarity of the phonetic feature matrices
    
    Will generate records of the form: ((phonemes:str, phonemes:str), LSH_bitwise_difference:int)
    """
    d = {}
    # rotate over each bit in the simhash
    for i in range(2*bits):
        def lsh(token):
            return token.simhash_rotate(rotations=i, n=n, bits=bits)
        for ngram in ngrams(sorted(tokens, key=lsh), n=window):
            # check each pairwise combination within the window
            for a, b in combinations(ngram, 2):
                key = tuple(sorted((a, b)))
                if key not in d:
                    token_a, token_b = key
                    d[key] = simdiff(lsh(a), lsh(b))
    yield from sorted(d.items(), key=itemgetter(1))

def compare(
    tokens,
    n=n,
    bits=hashsize,
    window=window
):
    """Return a datafram comparing pairs of Tokens sorted by phonemic similarity
    
    Datafram columns:
    "a": the first token of the pair
    "b": the second token of the pair
    "simhash difference (in bits)": the bitwise difference between the simhash of a and the simhash of b
    "sigmage(phonemic)": the phonemic similarity score for the pair (a, b) computed by: 1 - (difference/(2*bits)
    """
    try:
        get_ipython
        from tqdm.notebook import tqdm
    except:
        from tqdm import tqdm
    pairs = tqdm(
        list(ranked_pairs(tokens, n=n, bits=bits, window=window)),
        desc='ranking pairs by bitwise similarity',
        unit='pair'
    )
    return pd.DataFrame(
        {
            'a': a,
            'b': b,
            'simhash difference (in bits)': difference,
            'similarity score': f'{1.0 - (difference/(2*bits)):0.3}'
        } for ((a, b), difference) in pairs
    )
