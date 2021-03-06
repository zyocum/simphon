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
    Token(language='ell', graphemes='Σοφία', phonemes=['s', 'o', 'f', 'i', 'ɐ']),
    Token(language='ell', graphemes='Αλέξανδρος', phonemes=['ɐ', 'l', 'ɛ', 'k', 's', 'ɐ', 'n', 'ð', 'ɾ', 'o', 's']),
    Token(language='eng', graphemes='Adam', phonemes=['æ', 'd', 'ə', 'm']),
    Token(language='eng', graphemes='Alex', phonemes=['æ', 'l', 'ɪ', 'k', 's']),
    Token(language='eng', graphemes='Alexander', phonemes=['æ', 'l', 'ɪ', 'k', 'z', 'æ', 'n', 'd', 'ɚ']),
    Token(language='eng', graphemes='Alexis', phonemes=['ə', 'l', 'ɛ', 'k', 's', 'ɪ', 's']),
    Token(language='eng', graphemes='Amit', phonemes=['ɑ', 'm', 'i', 't']),
    Token(language='eng', graphemes='Andrew', phonemes=['æ', 'n', 'd', 'ɹ', 'u']),
    Token(language='eng', graphemes='Brad', phonemes=['b', 'ɹ', 'æ', 'd']),
    Token(language='eng', graphemes='Bradley', phonemes=['b', 'ɹ', 'æ', 'd', 'l', 'i']),
    Token(language='eng', graphemes='Brett', phonemes=['b', 'ɹ', 'ɛ', 't']),
    Token(language='eng', graphemes='Carl', phonemes=['k', 'ɑ', 'ɹ', 'l']),
    Token(language='eng', graphemes='Carlos', phonemes=['k', 'ɑ', 'ɹ', 'l', 'oʊ', 's']),
    Token(language='eng', graphemes='Catherine', phonemes=['k', 'æ', 'θ', 'ə', 'ɹ', 'ɪ', 'n']),
    Token(language='eng', graphemes='Charles', phonemes=['t̠ʃ', 'ɑ', 'ɹ', 'l', 'z']),
    Token(language='eng', graphemes='Drew', phonemes=['d', 'ɹ', 'u']),
    Token(language='eng', graphemes='Jennifer', phonemes=['d̠ʒ', 'ɛ', 'n', 'ə', 'f', 'ɚ']),
    Token(language='eng', graphemes='Jenny', phonemes=['d̠ʒ', 'ɛ', 'n', 'i']),
    Token(language='eng', graphemes='John', phonemes=['d̠ʒ', 'ɑ', 'n']),
    Token(language='eng', graphemes='Johnny', phonemes=['d̠ʒ', 'ɑ', 'n', 'i']),
    Token(language='eng', graphemes='Jonathan', phonemes=['d̠ʒ', 'ɑ', 'n', 'ə', 'θ', 'ə', 'n']),
    Token(language='eng', graphemes='Kat', phonemes=['k', 'æ', 't']),
    Token(language='eng', graphemes='Kathy', phonemes=['k', 'æ', 'θ', 'i']),
    Token(language='eng', graphemes='Katsuya', phonemes=['k', 'æ', 't', 's', 'uː', 'j', 'ə']),
    Token(language='eng', graphemes='Matt', phonemes=['m', 'æ', 't']),
    Token(language='eng', graphemes='Matthew', phonemes=['m', 'æ', 'θ', 'j', 'u']),
    Token(language='eng', graphemes='Michael', phonemes=['m', 'a', 'ɪ', 'k', 'ə', 'l']),
    Token(language='eng', graphemes='Mike', phonemes=['m', 'aɪ', 'k']),
    Token(language='eng', graphemes='Nate', phonemes=['n', 'eɪ', 't']),
    Token(language='eng', graphemes='Nathan', phonemes=['n', 'eɪ', 'θ', 'ə', 'n']),
    Token(language='eng', graphemes='Nathaniel', phonemes=['n', 'ə', 'θ', 'æ', 'n', 'j', 'ə', 'l']),
    Token(language='eng', graphemes='Nichole', phonemes=['n', 'ɪ', 'k', 'oʊ', 'l']),
    Token(language='eng', graphemes='Nick', phonemes=['n', 'ɪ', 'k']),
    Token(language='eng', graphemes='Phil', phonemes=['f', 'ɪ', 'l']),
    Token(language='eng', graphemes='Philip', phonemes=['f', 'ɪ', 'l', 'ɪ', 'p']),
    Token(language='eng', graphemes='Shinzo', phonemes=['ʃ', 'ɪ', 'n', 'z', 'oʊ']),
    Token(language='eng', graphemes='Smith', phonemes=['s', 'm', 'ɪ', 'θ']),
    Token(language='eng', graphemes='Sophia', phonemes=['s', 'oʊ', 'f', 'i', 'ə']),
    Token(language='eng', graphemes='Sophie', phonemes=['s', 'oʊ', 'f', 'i']),
    Token(language='eng', graphemes='Tsofit', phonemes=['s', 'oʊ', 'f', 'i', 't']),
    Token(language='eng', graphemes='Ty', phonemes=['t', 'aɪ']),
    Token(language='eng', graphemes='Tyler', phonemes=['t', 'aɪ', 'l', 'ə', 'ɹ']),
    Token(language='eng', graphemes='Xander', phonemes=['z', 'æ', 'n', 'd', 'ɚ']),
    Token(language='eng', graphemes='Zach', phonemes=['z', 'æ', 'k']),
    Token(language='eng', graphemes='Zachariah', phonemes=['z', 'æ', 'k', 'ə', 'ɹ', 'aɪ', 'ə']),
    Token(language='eng', graphemes='Zachary', phonemes=['z', 'æ', 'k', 'ə', 'ɹ', 'i']),
    Token(language='eng', graphemes='Zack', phonemes=['z', 'æ', 'k']),
    Token(language='fra', graphemes='Smith', phonemes=['s', 'm', 'i', 't']),
    Token(language='heb', graphemes='זך', phonemes=['z', 'a', 'k']),
    Token(language='heb', graphemes='עמת', phonemes=['a', 'm', 'ɪ', 't']),
    Token(language='heb', graphemes='יצח', phonemes=['j', 'i', 'ts', 'a', 'x']),
    Token(language='heb', graphemes='צוֹפִית', phonemes=['ts', 'o', 'f', 'i', 't']),
    Token(language='heb', graphemes='צוּפִית', phonemes=['ts', 'u', 'f', 'i', 't']),
    Token(language='heb', graphemes='זְכַרְיָה', phonemes=['z', 'ə̆', 'x', 'a', 'ʁ', 'j', 'a']),
    Token(language='hin', graphemes='आदम', phonemes=['ɑ', 'd̪', 'ə', 'm']),
    Token(language='hin', graphemes='अमित', phonemes=['aː', 'm', 'ɪ', 't̪']),
    Token(language='jpn', graphemes='カツヤ', phonemes=['k', 'a', 'tsː', 'ɯ', 'j', 'a']),
    Token(language='jpn', graphemes='シンゾ', phonemes=['s', 'i', 'n', 'd', 'z', 'ɔː']),
    Token(language='jpn', graphemes='スミス', phonemes=['s', 'ɯ', 'm', 'i', 's', 'ɯ']),
    Token(language='rus', graphemes='Алексей', phonemes=['a', 'lʲ', 'e', 'k', 'sʲ', 'e', 'j'])
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
