#!/usr/bin/env python3

"""Script with a CLI to show a simple demonstration of phonemic token matching.

The similarity of tokens is computed via simhashing of 2D matrices formed by
stacking PHOIBLE phoneme feature vectors for sequences of phonemes."""

import sys

import numpy as np
import pandas as pd

from dataclasses import dataclass
from functools import lru_cache
from functools import total_ordering
from lsh_utils import compare
from lsh_utils import hashes
from lsh_utils import Hashable_Ndarray
from lsh_utils import matrix_simhash
from lsh_utils import phoible

phoible_features = [
    'Phoneme',
    'Allophones',
    'Marginal',
    'SegmentClass',
    'tone',
    'stress',
    'syllabic',
    'short',
    'long',
    'consonantal',
    'sonorant',
    'continuant',
    'delayedRelease',
    'approximant',
    'tap',
    'trill',
    'nasal',
    'lateral',
    'labial',
    'round',
    'labiodental',
    'coronal',
    'anterior',
    'distributed',
    'strident',
    'dorsal',
    'high',
    'low',
    'front',
    'back',
    'tense',
    'retractedTongueRoot',
    'advancedTongueRoot',
    'periodicGlottalSource',
    'epilaryngealSource',
    'spreadGlottis',
    'constrictedGlottis'
]

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
        """Get this Token's simhash"""
        matrix = Token.phonemes_matrix(self.phonemes, language=self.language)
        return matrix_simhash(matrix, n=n, bits=bits)
    
    @lru_cache
    def simhash_rotate(self, rotations=1, n=2, bits=128):
        """Get this Token's simhash after bitwise a number of bitwise rotations (this is cached)"""
        simhash = self.simhash(n=n, bits=bits)
        actual_bitwidth = simhash.bit_length()
        rotations %= actual_bitwidth
        if rotations < 1:
            return simhash
        mask = (2 ** actual_bitwidth) - 1
        simhash &= mask
        return (simhash >> rotations) | ((simhash << (actual_bitwidth - rotations) & mask))
    
    @staticmethod
    def phoneme_vector(phoneme, language='eng'):
        """Get a discrete vector representation of a phoneme in the given language from PHOIBLE's data"""
        data = phoible[phoible['ISO6393'] == language]
        try:
            vector = data[data['Phoneme'] == phoneme].iloc[0][phoible_features]
            return vector.values.astype(str)
        except IndexError:
            print(f'Failed to find features for {phoneme!r} in {language!r}', file=sys.stderr)
            sys.exit(1)
    
    @staticmethod
    def phonemes_matrix(phonemes, language='eng'):
        """Get a hashable np.ndarray subclass containing a 2D PHOIBLE feature matrix representation of the given phonemes"""
        return Hashable_Ndarray(np.stack([Token.phoneme_vector(p, language=language) for p in phonemes]))
    
    def as_feature_matrix(self):
        """Get a pd.DataFrame representation of the PHOIBLE features of this Token's phonemes"""
        matrix = self.phonemes_matrix(self.phonemes, self.language)
        return pd.DataFrame(matrix.T, index=phoible_features, columns=self.phonemes)

tokens = [
    Token(language='ell', graphemes='??????????', phonemes=['s', 'o', 'f', 'i', '??']),
    Token(language='ell', graphemes='????????????????????', phonemes=['??', 'l', '??', 'k', 's', '??', 'n', '??', '??', 'o', 's']),
    Token(language='eng', graphemes='Adam', phonemes=['??', 'd', '??', 'm']),
    Token(language='eng', graphemes='Alex', phonemes=['??', 'l', '??', 'k', 's']),
    Token(language='eng', graphemes='Alexander', phonemes=['??', 'l', '??', 'k', 'z', '??', 'n', 'd', '??']),
    Token(language='eng', graphemes='Alexis', phonemes=['??', 'l', '??', 'k', 's', '??', 's']),
    Token(language='eng', graphemes='Amit', phonemes=['??', 'm', 'i', 't']),
    Token(language='eng', graphemes='Andrew', phonemes=['??', 'n', 'd', '??', 'u']),
    Token(language='eng', graphemes='Brad', phonemes=['b', '??', '??', 'd']),
    Token(language='eng', graphemes='Bradley', phonemes=['b', '??', '??', 'd', 'l', 'i']),
    Token(language='eng', graphemes='Brett', phonemes=['b', '??', '??', 't']),
    Token(language='eng', graphemes='Carl', phonemes=['k', '??', '??', 'l']),
    Token(language='eng', graphemes='Carlos', phonemes=['k', '??', '??', 'l', 'o??', 's']),
    Token(language='eng', graphemes='Catherine', phonemes=['k', '??', '??', '??', '??', '??', 'n']),
    Token(language='eng', graphemes='Charles', phonemes=['t????', '??', '??', 'l', 'z']),
    Token(language='eng', graphemes='Drew', phonemes=['d', '??', 'u']),
    Token(language='eng', graphemes='Jennifer', phonemes=['d????', '??', 'n', '??', 'f', '??']),
    Token(language='eng', graphemes='Jenny', phonemes=['d????', '??', 'n', 'i']),
    Token(language='eng', graphemes='John', phonemes=['d????', '??', 'n']),
    Token(language='eng', graphemes='Johnny', phonemes=['d????', '??', 'n', 'i']),
    Token(language='eng', graphemes='Jonathan', phonemes=['d????', '??', 'n', '??', '??', '??', 'n']),
    Token(language='eng', graphemes='Kat', phonemes=['k', '??', 't']),
    Token(language='eng', graphemes='Kathy', phonemes=['k', '??', '??', 'i']),
    Token(language='eng', graphemes='Katsuya', phonemes=['k', '??', 't', 's', 'u??', 'j', '??']),
    Token(language='eng', graphemes='Matt', phonemes=['m', '??', 't']),
    Token(language='eng', graphemes='Matthew', phonemes=['m', '??', '??', 'j', 'u']),
    Token(language='eng', graphemes='Michael', phonemes=['m', 'a', '??', 'k', '??', 'l']),
    Token(language='eng', graphemes='Mike', phonemes=['m', 'a??', 'k']),
    Token(language='eng', graphemes='Nate', phonemes=['n', 'e??', 't']),
    Token(language='eng', graphemes='Nathan', phonemes=['n', 'e??', '??', '??', 'n']),
    Token(language='eng', graphemes='Nathaniel', phonemes=['n', '??', '??', '??', 'n', 'j', '??', 'l']),
    Token(language='eng', graphemes='Nichole', phonemes=['n', '??', 'k', 'o??', 'l']),
    Token(language='eng', graphemes='Nick', phonemes=['n', '??', 'k']),
    Token(language='eng', graphemes='Phil', phonemes=['f', '??', 'l']),
    Token(language='eng', graphemes='Philip', phonemes=['f', '??', 'l', '??', 'p']),
    Token(language='eng', graphemes='Shinzo', phonemes=['??', '??', 'n', 'z', 'o??']),
    Token(language='eng', graphemes='Smith', phonemes=['s', 'm', '??', '??']),
    Token(language='eng', graphemes='Sophia', phonemes=['s', 'o??', 'f', 'i', '??']),
    Token(language='eng', graphemes='Sophie', phonemes=['s', 'o??', 'f', 'i']),
    Token(language='eng', graphemes='Tsofit', phonemes=['s', 'o??', 'f', 'i', 't']),
    Token(language='eng', graphemes='Ty', phonemes=['t', 'a??']),
    Token(language='eng', graphemes='Tyler', phonemes=['t', 'a??', 'l', '??', '??']),
    Token(language='eng', graphemes='Xander', phonemes=['z', '??', 'n', 'd', '??']),
    Token(language='eng', graphemes='Zach', phonemes=['z', '??', 'k']),
    Token(language='eng', graphemes='Zachariah', phonemes=['z', '??', 'k', '??', '??', 'a??', '??']),
    Token(language='eng', graphemes='Zachary', phonemes=['z', '??', 'k', '??', '??', 'i']),
    Token(language='eng', graphemes='Zack', phonemes=['z', '??', 'k']),
    Token(language='fra', graphemes='Smith', phonemes=['s', 'm', 'i', 't']),
    Token(language='heb', graphemes='????', phonemes=['z', 'a', 'k']),
    Token(language='heb', graphemes='??????', phonemes=['a', 'm', '??', 't']),
    Token(language='heb', graphemes='??????', phonemes=['j', 'i', 'ts', 'a', 'x']),
    Token(language='heb', graphemes='??????????????', phonemes=['ts', 'o', 'f', 'i', 't']),
    Token(language='heb', graphemes='??????????????', phonemes=['ts', 'u', 'f', 'i', 't']),
    Token(language='heb', graphemes='??????????????????', phonemes=['z', '????', 'x', 'a', '??', 'j', 'a']),
    Token(language='hin', graphemes='?????????', phonemes=['??', 'd??', '??', 'm']),
    Token(language='hin', graphemes='????????????', phonemes=['a??', 'm', '??', 't??']),
    Token(language='jpn', graphemes='?????????', phonemes=['k', 'a', 'ts??', '??', 'j', 'a']),
    Token(language='jpn', graphemes='?????????', phonemes=['s', 'i', 'n', 'd', 'z', '????']),
    Token(language='jpn', graphemes='?????????', phonemes=['s', '??', 'm', 'i', 's', '??']),
    Token(language='rus', graphemes='??????????????', phonemes=['a', 'l??', 'e', 'k', 's??', 'e', 'j'])
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
