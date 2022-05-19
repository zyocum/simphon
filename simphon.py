#!/usr/bin/env python3

"""Script with a CLI to show a simple demonstration of phonemic token matching.

The similarity of tokens is computed via simhashing of 2D matrices formed by
stacking PHOIBLE phoneme feature fectors for sequences of phonemes."""

import sys

import numpy as np
import pandas as pd

from dataclasses import dataclass
from functools import lru_cache
from functools import total_ordering
from lsh_utils import compare
from lsh_utils import hashes
from lsh_utils import matrix_simhash
from lsh_utils import phoible

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
    def simhash(self, n=2, bits=128):
        matrix = Token.phonemes_matrix(self.phonemes, language=self.language)
        return matrix_simhash(matrix, n=n, bits=bits)
    
    @lru_cache
    def simhash_rotate(self, rotations=1, n=2, bits=128):
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
        except IndexError:
            print(f'Failed to find features for {phoneme!r} in {language!r}', file=sys.stderr)
            sys.exit(1)
    
    @staticmethod
    def phonemes_matrix(phonemes, language='eng'):
        return np.stack([Token.phoneme_vector(p, language=language) for p in phonemes])
    
    def as_feature_matrix(self):
        matrix = self.phonemes_matrix(self.phonemes, self.language)
        return pd.DataFrame(matrix.T, index=phoible.columns[12:], columns=self.phonemes)

tokens = [
    Token(language='eng', graphemes='Alex', phonemes=['æ','l','ə','k','s']),
    Token(language='eng', graphemes='Alexander', phonemes=['æ','l','ə','k','z','æ','n','d','ɚ']),
    Token(language='eng', graphemes='Alexi', phonemes=['ə','l','ɛ','k','s','i']),
    Token(language='eng', graphemes='Alexis', phonemes=['ə','l','ɛ','k','s','ɪ','s']),
    Token(language='eng', graphemes='Andrew', phonemes=['æ','n','d','ɹ','u']),
    Token(language='eng', graphemes='Brad', phonemes=['b','ɹ','æ','d']),
    Token(language='eng', graphemes='Bradley', phonemes=['b','ɹ','æ','d','l','i']),
    Token(language='eng', graphemes='Brett', phonemes=['b','ɹ','ɛ','t']),
    Token(language='eng', graphemes='Carl', phonemes=['k','ɑ','ɹ','l']),
    Token(language='eng', graphemes='Carlos', phonemes=['k','ɑ','ɹ','l','oʊ','s']),
    Token(language='eng', graphemes='Catherine', phonemes=['k','æ','θ','ə','ɹ','ə','n']),
    Token(language='eng', graphemes='Catherine', phonemes=['k','æ','θ','ə','ɹ','ɪ','n']),
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
    Token(language='eng', graphemes='Tyler', phonemes=['t','aɪ','l','ə','ɹ']),
    Token(language='eng', graphemes='Xander', phonemes=['z','æ','n','d','ɚ']),
    Token(language='eng', graphemes='Zach', phonemes=['z','æ','k']),
    Token(language='eng', graphemes='Zachary', phonemes=['z','æ','k','ə','ɹ','i']),
    Token(language='eng', graphemes='Zak', phonemes=['z','æ','k']),
]

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
        choices=hashes,
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
            tokens,
            n=args.n_grams_size,
            bits=args.bits,
            window=args.window
        ).to_csv(sep='\t')
    )
