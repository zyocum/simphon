#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
import mmh3

from tqdm import tqdm
from functools import lru_cache
from functools import total_ordering
from itertools import combinations
from numpy.lib.stride_tricks import sliding_window_view
from operator import itemgetter
from pathlib import Path

HASHES = {
    32: mmh3.hash, # 32 bit murmurhash
    64: lambda x: mmh3.hash64(x)[0], # mmh3.hash64 returns two 64 bit hashes, so we take the first one
    128: mmh3.hash128 # 128 bit murmurhash
}

HASHSIZE = 64

def ngrams(iterable, n=1):
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

def segment_simhash(m, n=2, bits=HASHSIZE):
    """Compute a simhash over the bytes of the rows in a matrix"""
    hashf = HASHES[bits]
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

def stride_simhash(m, n=2, bits=HASHSIZE):
    hashf = HASHES[bits]
    lsh = np.zeros(bits)
    window_shape = (n, n)
    if m.shape < window_shape: # too small
        return 0
    for i, axis in enumerate(sliding_window_view(m, window_shape=window_shape)):
        for view in axis:
            #print(f'{i}:{"".join(view.flatten())}', end='\n')
            for j in range(bits):
                data = int.to_bytes(i, length=1, byteorder='big') + view.tobytes()
                if hashf(data) & (1 << j):
                    lsh[j] += 1
                else:
                    lsh[j] -= 1
    return sum(int(b > 0) << i for (i, b) in enumerate(reversed(lsh)))

def matrix_simhash(m, n=2, bits=HASHSIZE):
    """Compute a simhash by XORing simhashes of the rows and columns of a phoneme matrix
    and also a simhash based on a stride-based sliding window over the phoneme matrix"""
    simhash = segment_simhash(m, n=n, bits=bits)
    simhash ^= segment_simhash(m.T, n=n, bits=bits)
    simhash ^= stride_simhash(m, n=n, bits=bits) << bits
    return simhash

def rotate(n, rotations=1, width=HASHSIZE):
    """Bitwise rotate an int.

    bin(rotate(1, rotations=0))  ->                                '0b1'
    bin(rotate(1, rotations=1))  -> '0b10000000000000000000000000000000'
    bin(rotate(1, rotations=2))  ->  '0b1000000000000000000000000000000'
    bin(rotate(1, rotations=32)) ->                                '0b1'
    bin(rotate(1, rotations=31)) ->                               '0b10'
    bin(rotate(1, rotations=-1)) ->                               '0b10'
    bin(rotate(1, rotations=1, width=8)) ->                 '0b10000000'
    bin(rotate(1, rotations=8, width=8)) ->                        '0b1'

    """
    width = max(n.bit_length(), width)
    rotations %= width
    if rotations < 1:
        return n
    mask = (2 ** width) - 1
    n &= mask
    return (n >> rotations) | ((n << (width - rotations) & mask))

def simdiff(a, b):
    """Compute the bitwise difference between two simhashes"""
    bits = max(a.bit_length(), b.bit_length())
    if bits < 1:
        raise ValueError(f'need at least 1 bit to compute bitwise simhash difference (bits={bits})')
    xor = a ^ b
    difference = sum(((xor & (1 << i)) > 0) for i in range(bits))
    return difference

def load_phoible(path='phoible.csv', cache=False):
    if not(cache) or not(Path(path).is_file()):
        phoible = pd.read_csv('https://raw.githubusercontent.com/phoible/dev/master/data/phoible.csv', low_memory=False)
        with open(path, mode='w') as f:
            print(phoible.to_csv(), file=f)
    return pd.read_csv(path, low_memory=False)

phoible = load_phoible()

def phonemes2matrix(phonemes, language='eng'):
    """Get a discrete matrix representation of a sequence of phonemes in the given language from PHOIBLE's data"""
    return np.stack([phoneme2vec(p, language=language) for p in phonemes])

from dataclasses import dataclass
from dataclasses import field

@dataclass
@total_ordering
class Token:
    language: str
    graphemes: str
    phonemes: tuple
    
    def __str__(self):
        return f'({self.language}) {self.graphemes} /{" ".join(self.phonemes)}/'
    
    def __hash__(self):
        return hash((self.language, self.graphemes, ' '.join(self.phonemes)))
    
    def __lt__(self, other):
        return (self.language, self.graphemes, ' '.join(self.phonemes)) < \
            (other.language, other.graphemes, ' '.join(other.phonemes))
    
    @lru_cache
    def simhash(self, n=2, bits=HASHSIZE):
        matrix = Token.phonemes_matrix(self.phonemes, language=self.language)
        return matrix_simhash(matrix, n=n, bits=bits)
    
    @lru_cache
    def simhash_rotate(self, rotations=1, n=2, bits=HASHSIZE):
        r = self.simhash(n=n, bits=bits)
        width = r.bit_length()
        rotations %= width
        if rotations < 1:
            return r
        mask = (2 ** width) - 1
        r &= mask
        return (r >> rotations) | ((r << (width - rotations) & mask))
    
    @staticmethod
    def phoneme_vector(phoneme, language='eng'):
        """Get a discrete vector representation of a phoneme in the given language from PHOIBLE's data"""
        data = phoible[phoible['ISO6393'] == language]
        try:
            vector = data[data['Phoneme'] == phoneme].iloc[0].values[12:]
            return vector
        except IndexError as e:
            print(f'Failed to find features for {phoneme!r} in {language!r}', file=sys.stderr)
            sys.exit(1)
    
    @staticmethod
    def phonemes_matrix(phonemes, language='eng'):
        return np.stack([Token.phoneme_vector(p, language=language) for p in phonemes])
    
items = [
    Token(language='eng', graphemes='Alex', phonemes=['æ','l','ə','k','s']),
    Token(language='eng', graphemes='Alexander', phonemes=['æ','l','ə','z','æ','n','d','ɚ']),
    Token(language='eng', graphemes='Alexi', phonemes=['ə','l','ɛ','k','s','i']),
    Token(language='eng', graphemes='Alexis', phonemes=['ə','l','ɛ','k','s','ɪ','s']),
    Token(language='eng', graphemes='Andrew', phonemes=['æ','n','d','r','u']),
    Token(language='eng', graphemes='Brad', phonemes=['b','ɹ','æ','d']),
    Token(language='eng', graphemes='Bradley', phonemes=['b','ɹ','æ','d','l','i']),
    Token(language='eng', graphemes='Brett', phonemes=['b','ɹ','ɛ','t']),
    Token(language='eng', graphemes='Carl', phonemes=['k','ɑ','ɹ','l']),
    Token(language='eng', graphemes='Carlos', phonemes=['k','ɑ','ɹ','l','oʊ','s']),
    Token(language='eng', graphemes='Catherine', phonemes=['k','æ','θ','ə','r','ə','n']),
    Token(language='eng', graphemes='Catherine', phonemes=['k','æ','θ','ə','r','ɪ','n']),
    Token(language='eng', graphemes='Charles' , phonemes=['t̠ʃ','ɑ','ɹ','l','z']),
    Token(language='eng', graphemes='Drew', phonemes=['d','ɹ','u']),
    Token(language='eng', graphemes='Jennifer', phonemes=['d̠ʒ','ɛ','n','ə','f','ɚ']),
    Token(language='eng', graphemes='Jenny', phonemes=['d̠ʒ','ɛ','n','i']),
    Token(language='eng', graphemes='John', phonemes=['d̠ʒ','ɑ','n']),
    Token(language='eng', graphemes='Johnny', phonemes=['d̠ʒ','ɑ','n','i']),
    Token(language='eng', graphemes='Jonathan', phonemes=['d̠ʒ','ɑ','n','ə','θ','ə','n']),
    Token(language='eng', graphemes='Kat', phonemes=['k','æ','t']),
    Token(language='eng', graphemes='Kathy', phonemes=['k','æ','θ','i']),
    Token(language='eng', graphemes='Matt', phonemes=['m','æ','t']),
    Token(language='eng', graphemes='Matthew', phonemes=['m','æ','θ','j','u']),
    Token(language='eng', graphemes='Michael', phonemes=['m','a','ɪ','k','ə','l']),
    Token(language='eng', graphemes='Mike', phonemes=['m','aɪ','k']),
    Token(language='eng', graphemes='Nate', phonemes=['n','eɪ','t']),
    Token(language='eng', graphemes='Nathan', phonemes=['n','eɪ','θ','ə','n']),
    Token(language='eng', graphemes='Nathaniel', phonemes=['n','ə','θ','æ','n','j','ə','l']),
    Token(language='eng', graphemes='Nichole', phonemes=['n','ɪ','k','oʊ','l']),
    Token(language='eng', graphemes='Nick', phonemes=['n','ɪ','k']),
    Token(language='eng', graphemes='Phil', phonemes=['f','ɪ','l']),
    Token(language='eng', graphemes='Philip', phonemes=['f','ɪ','l','ɪ','p']),
    Token(language='eng', graphemes='Ty', phonemes=['t','aɪ']),
    Token(language='eng', graphemes='Tyler', phonemes=['t','aɪ','l','ə','r']),
    Token(language='eng', graphemes='Xander', phonemes=['z','æ','n','d','ɚ']),
    Token(language='eng', graphemes='Zach', phonemes=['z','æ','k']),
    Token(language='eng', graphemes='Zak', phonemes=['z','æ','k']),
    Token(language='eng', graphemes='Zachary', phonemes=['z','æ','k','ə','r','i']),
]

def ranked_pairs(tokens, n=2, bits=HASHSIZE, window=8):
    """Rank pairs of phonemic sequences sorted by LSH similarity of the phonetic feature matrices
    
    Will generate records of the form: ((phonemes:str, phonemes:str), LSH_bitwise_difference:int)
    """
    d = {}
    for i in range(bits):
        def lsh(token):
            return token.simhash_rotate(rotations=i, n=n, bits=bits)
        for ngram in ngrams(sorted(tokens, key=lsh), n=window):
            for a, b in combinations(ngram, 2):
                key = tuple(sorted((a, b)))
                if key not in d:
                    token_a, token_b = key
                    d[key] = simdiff(lsh(a), lsh(b))
    yield from sorted(d.items(), key=itemgetter(1))

def compare(
        items,
        n=3, # defines size of n-grams and n x n sliding-window features
        bits=128, # defines underlying LSH bit size (actual size will be 2*bits)
        window=10 # defines size of window in which to consider pairs for computing pair-wise distances
):
    pairs = tqdm(
        list(ranked_pairs(items, n=n, bits=bits, window=window)),
        desc='ranking pairs by bitwise distances',
        unit='pair'
    )
    return pd.DataFrame(
        {
            'a': a,
            'b': b,
            'simhash difference (in bits)': difference,
            '$$\sigma_{phonemic}$$': f'{1.0 - (difference/(2*bits)):0.3}'
        } for ((a, b), difference) in pairs
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    parser.add_argument(
        '-n', '--n-grams-size',
        type=int,
        default=3,
        help='size n for n-grams',
    )
    parser.add_argument(
        '-b', '--bits',
        type=int,
        default=128,
        choices=HASHES,
        help='LSH bit size'
    )
    parser.add_argument(
        '-w', '--window',
        type=int,
        default=10,
        help='sliding window size for comparing LSH bitwise differences',
    )
    args = parser.parse_args()
    print(
        compare(
            items,
            n=args.n_grams_size,
            bits=args.bits,
            window=args.window
        ).to_csv(sep='\t')
    )
