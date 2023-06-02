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
from lsh_utils import search
from lsh_utils import Hashable_Ndarray
from lsh_utils import matrix_simhash
from lsh_utils import phoible
from lsh_utils import DEFAULT_HASHSIZE
from lsh_utils import SIMHASH_BITS
from lsh_utils import DEFAULT_SEED
from lsh_utils import DEFAULT_WINDOW_SIZE
from lsh_utils import CACHE_SIZE
from typing import ClassVar

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

@lru_cache(maxsize=CACHE_SIZE)
def phonemic_inventory(language):
    """Get a DataFrame containing the phonemes for the given language"""
    return pd.concat(
        (
            phoible[phoible['ISO6393'] == language],
            # add special sentinel "|"to the phonemic inventory to represent word boundaries
            # th represent word boundaries (start=^, end=$)
            pd.DataFrame([
                dict({key: '-' for key in phoible.columns}, **{'Phoneme': '|'}),
            ]),
        ),
        ignore_index=True
    )

@lru_cache(maxsize=CACHE_SIZE)
def phoneme_vector(phoneme, language='eng'):
    """Get a discrete vector representation of a phoneme in the given language from PHOIBLE's data"""
    data = phonemic_inventory(language)
    try:
        vector = data[data['Phoneme'] == phoneme].iloc[0][phoible_features]
        return vector.values.astype(str)
    except IndexError:
        print(f'Failed to find features for {phoneme=!r} in {language=!r}', file=sys.stderr)
        sys.exit(1)

@dataclass
@total_ordering
class Token:
    language: str
    graphemes: str
    phonemes: tuple
    hashsize: ClassVar[int] = DEFAULT_HASHSIZE
    seed: ClassVar[int] = DEFAULT_SEED
    
    def __str__(self):
        return f'({self.language}) 〈{self.graphemes}〉 /{" ".join(self.phonemes)}/'
    
    def __hash__(self):
        return hash((self.language, self.graphemes, ' '.join(self.phonemes)))
    
    def __lt__(self, other):
        return (self.language, self.graphemes, ' '.join(self.phonemes)) < \
            (other.language, other.graphemes, ' '.join(other.phonemes))
    
    @lru_cache(maxsize=CACHE_SIZE)
    def simhash(self):
        """Get this Token's simhash"""
        matrix = self.phonemes_matrix(self.phonemes, language=self.language)
        return matrix_simhash(matrix, hashsize=self.hashsize, seed=self.seed)
    
    @lru_cache(maxsize=CACHE_SIZE)
    def phonemes_matrix(self, phonemes, language='eng'):
        """Get a hashable np.ndarray subclass containing a 2D PHOIBLE feature matrix representation of the given phonemes"""
        return Hashable_Ndarray(np.stack([phoneme_vector(p, language=language) for p in phonemes]))
    
    def as_feature_matrix(self):
        """Get a pd.DataFrame representation of the PHOIBLE features of this Token's phonemes"""
        matrix = self.phonemes_matrix(self.phonemes, self.language)
        return pd.DataFrame(matrix.T, index=phoible_features, columns=self.phonemes)

tokens = [
    Token(language='ell', graphemes='Σοφία', phonemes=('|', 's', 'o', 'f', 'i', 'ɐ', '|')),
    Token(language='ell', graphemes='Αλέξανδρος', phonemes=('|', 'ɐ', 'l', 'ɛ', 'k', 's', 'ɐ', 'n', 'ð', 'ɾ', 'o', 's', '|')),
    Token(language='eng', graphemes='Adam', phonemes=('|', 'æ', 'd', 'ə', 'm', '|')),
    Token(language='eng', graphemes='Alex', phonemes=('|', 'æ', 'l', 'ɪ', 'k', 's', '|')),
    Token(language='eng', graphemes='Alexander', phonemes=('|', 'æ', 'l', 'ɪ', 'k', 'z', 'æ', 'n', 'd', 'ɚ', '|')),
    Token(language='eng', graphemes='Alexis', phonemes=('|', 'ə', 'l', 'ɛ', 'k', 's', 'ɪ', 's', '|')),
    Token(language='eng', graphemes='Amit', phonemes=('|', 'ɑ', 'm', 'i', 't', '|')),
    Token(language='eng', graphemes='Andrew', phonemes=('|', 'æ', 'n', 'd', 'ɹ', 'u', '|')),
    Token(language='eng', graphemes='Brad', phonemes=('|', 'b', 'ɹ', 'æ', 'd', '|')),
    Token(language='eng', graphemes='Bradley', phonemes=('|', 'b', 'ɹ', 'æ', 'd', 'l', 'i', '|')),
    Token(language='eng', graphemes='Brett', phonemes=('|', 'b', 'ɹ', 'ɛ', 't', '|')),
    Token(language='eng', graphemes='Carl', phonemes=('|', 'k', 'ɑ', 'ɹ', 'l', '|')),
    Token(language='eng', graphemes='Carlos', phonemes=('|', 'k', 'ɑ', 'ɹ', 'l', 'oʊ', 's', '|')),
    Token(language='eng', graphemes='Catherine', phonemes=('|', 'k', 'æ', 'θ', 'ə', 'ɹ', 'ɪ', 'n', '|')),
    Token(language='eng', graphemes='Charles', phonemes=('|', 't̠ʃ', 'ɑ', 'ɹ', 'l', 'z', '|')),
    Token(language='eng', graphemes='Drew', phonemes=('|', 'd', 'ɹ', 'u', '|')),
    Token(language='eng', graphemes='Jennifer', phonemes=('|', 'd̠ʒ', 'ɛ', 'n', 'ə', 'f', 'ɚ', '|')),
    Token(language='eng', graphemes='Jenny', phonemes=('|', 'd̠ʒ', 'ɛ', 'n', 'i', '|')),
    Token(language='eng', graphemes='John', phonemes=('|', 'd̠ʒ', 'ɑ', 'n', '|')),
    Token(language='eng', graphemes='Johnny', phonemes=('|', 'd̠ʒ', 'ɑ', 'n', 'i', '|')),
    Token(language='eng', graphemes='Jonathan', phonemes=('|', 'd̠ʒ', 'ɑ', 'n', 'ə', 'θ', 'ə', 'n', '|')),
    Token(language='eng', graphemes='Kat', phonemes=('|', 'k', 'æ', 't', '|')),
    Token(language='eng', graphemes='Kathy', phonemes=('|', 'k', 'æ', 'θ', 'i', '|')),
    Token(language='eng', graphemes='Katsuya', phonemes=('|', 'k', 'æ', 't', 's', 'uː', 'j', 'ə', '|')),
    Token(language='eng', graphemes='Matt', phonemes=('|', 'm', 'æ', 't', '|')),
    Token(language='eng', graphemes='Matthew', phonemes=('|', 'm', 'æ', 'θ', 'j', 'u', '|')),
    Token(language='eng', graphemes='Michael', phonemes=('|', 'm', 'a', 'ɪ', 'k', 'ə', 'l', '|')),
    Token(language='eng', graphemes='Mike', phonemes=('|', 'm', 'aɪ', 'k', '|')),
    Token(language='eng', graphemes='Nate', phonemes=('|', 'n', 'eɪ', 't', '|')),
    Token(language='eng', graphemes='Nathan', phonemes=('|', 'n', 'eɪ', 'θ', 'ə', 'n', '|')),
    Token(language='eng', graphemes='Nathaniel', phonemes=('|', 'n', 'ə', 'θ', 'æ', 'n', 'j', 'ə', 'l', '|')),
    Token(language='eng', graphemes='Nichole', phonemes=('|', 'n', 'ɪ', 'k', 'oʊ', 'l', '|')),
    Token(language='eng', graphemes='Nick', phonemes=('|', 'n', 'ɪ', 'k', '|')),
    Token(language='eng', graphemes='Phil', phonemes=('|', 'f', 'ɪ', 'l', '|')),
    Token(language='eng', graphemes='Philip', phonemes=('|', 'f', 'ɪ', 'l', 'ɪ', 'p', '|')),
    Token(language='eng', graphemes='Shinzo', phonemes=('|', 'ʃ', 'ɪ', 'n', 'z', 'oʊ', '|')),
    Token(language='eng', graphemes='Smith', phonemes=('|', 's', 'm', 'ɪ', 'θ', '|')),
    Token(language='eng', graphemes='Sophia', phonemes=('|', 's', 'oʊ', 'f', 'i', 'ə', '|')),
    Token(language='eng', graphemes='Sophie', phonemes=('|', 's', 'oʊ', 'f', 'i', '|')),
    Token(language='eng', graphemes='Tsofit', phonemes=('|', 's', 'oʊ', 'f', 'i', 't', '|')),
    Token(language='eng', graphemes='Ty', phonemes=('|', 't', 'aɪ', '|')),
    Token(language='eng', graphemes='Tyler', phonemes=('|', 't', 'aɪ', 'l', 'ə', 'ɹ', '|')),
    Token(language='eng', graphemes='Xander', phonemes=('|', 'z', 'æ', 'n', 'd', 'ɚ', '|')),
    Token(language='eng', graphemes='Zach', phonemes=('|', 'z', 'æ', 'k', '|')),
    Token(language='eng', graphemes='Zachariah', phonemes=('|', 'z', 'æ', 'k', 'ə', 'ɹ', 'aɪ', 'ə', '|')),
    Token(language='eng', graphemes='Zachary', phonemes=('|', 'z', 'æ', 'k', 'ə', 'ɹ', 'i', '|')),
    Token(language='eng', graphemes='Zack', phonemes=('|', 'z', 'æ', 'k', '|')),
    Token(language='fra', graphemes='Smith', phonemes=('|', 's', 'm', 'i', 't', '|')),
    Token(language='fra', graphemes='Sophie', phonemes=('|', 's', 'ɔ', 'f', 'i', '|')),
    Token(language='fra', graphemes='Sofia', phonemes=('|', 's', 'ɔ', 'f', 'j', 'a', '|')),
    Token(language='heb', graphemes='זך', phonemes=('|', 'z', 'a', 'k', '|')),
    Token(language='heb', graphemes='עמת', phonemes=('|', 'a', 'm', 'ɪ', 't', '|')),
    Token(language='heb', graphemes='יצח', phonemes=('|', 'j', 'i', 'ts', 'a', 'x', '|')),
    Token(language='heb', graphemes='צוֹפִית', phonemes=('|', 'ts', 'o', 'f', 'i', 't', '|')),
    Token(language='heb', graphemes='צוּפִית', phonemes=('|', 'ts', 'u', 'f', 'i', 't', '|')),
    Token(language='heb', graphemes='זְכַרְיָה', phonemes=('|', 'z', 'ə̆', 'x', 'a', 'ʁ', 'j', 'a', '|')),
    Token(language='hin', graphemes='आदम', phonemes=('|', 'ɑ', 'd̪', 'ə', 'm', '|')),
    Token(language='hin', graphemes='अमित', phonemes=('|', 'aː', 'm', 'ɪ', 't̪', '|')),
    Token(language='jpn', graphemes='カツヤ', phonemes=('|', 'k', 'a', 'tsː', 'ɯ', 'j', 'a', '|')),
    Token(language='jpn', graphemes='シンゾ', phonemes=('|', 's', 'i', 'n', 'd', 'z', 'ɔː', '|')),
    Token(language='jpn', graphemes='スミス', phonemes=('|', 's', 'ɯ', 'm', 'i', 's', 'ɯ', '|')),
    Token(language='jpn', graphemes='まさよし', phonemes=('|', 'm', 'a', 's', 'a', 'j', 'o', 's', 'i', '|')),
    Token(language='jpn', graphemes='ただよし', phonemes=('|', 't', 'a', 'd', 'a', 'j', 'o', 's', 'i', '|')),
    Token(language='jpn', graphemes='かつら', phonemes=('|', 'k', 'a', 'tsː', 'ɯ', 'd̠', 'a', '|')),
    Token(language='jpn', graphemes='カツラ', phonemes=('|', 'k', 'a', 'tsː', 'ɯ', 'd̠', 'a', '|')),
    Token(language='rus', graphemes='Алексей', phonemes=('|', 'a', 'lʲ', 'e', 'k', 'sʲ', 'e', 'j', '|'))
]

objs = (
    {"language": "eng", "graphemes": "Adam Bernstein", "phonemes": "/ædəm bɛrnstin/", "segments": [["æ", "d", "ə", "m"], ["b", "ɛ", "r", "n", "s", "t", "i", "n"]]},
    {"language": "eng", "graphemes": "Adam Bernstein", "phonemes": "/ædəm bɛrnstaɪn/", "segments": [["æ", "d", "ə", "m"], ["b", "ɛ", "r", "n", "s", "t", "aɪ", "n"]]},
    {"language": "eng", "graphemes": "Bernstein, Adam", "phonemes": "/bɛrnstin ædəm/", "segments": [["b", "ɛ", "r", "n", "s", "t", "i", "n"], ["æ", "d", "ə", "m"]]},
    {"language": "eng", "graphemes": "Alison Schapker", "phonemes": "/æləsn ʃæpkɹ/", "segments": [["æ", "l", "ə", "s", "n"], ["ʃ", "æ", "p", "k", "ɹ"]]},
    {"language": "eng", "graphemes": "Alli Rock", "phonemes": "/æli ɹɑk/", "segments": [["æ", "l", "i"], ["ɹ", "ɑ", "k"]]},
    {"language": "eng", "graphemes": "Amanda Marsalis", "phonemes": "/əmændə mɑrsəlɪs/", "segments": [["ə", "m", "æ", "n", "d", "ə"], ["m", "ɑ", "r", "s", "ə", "l", "ɪ", "s"]]},
    {"language": "eng", "graphemes": "Andrew Guest", "phonemes": "/ændɹu ɡɛst/", "segments": [["æ", "n", "d", "ɹ", "u"], ["ɡ", "ɛ", "s", "t"]]},
    {"language": "eng", "graphemes": "Andrew Seklir", "phonemes": "/ændɹu sɛkliɹ/", "segments": [["æ", "n", "d", "ɹ", "u"], ["s", "ɛ", "k", "l", "ɪ", "ɹ"]]},
    {"language": "eng", "graphemes": "Andrew Seklir", "phonemes": "/ændɹu sɛkləɹ/", "segments": [["æ", "n", "d", "ɹ", "u"], ["s", "ɛ", "k", "l", "ə", "ɹ"]]},
    {"language": "eng", "graphemes": "Anna Foerster", "phonemes": "/ænə fɔɹstɹ/", "segments": [["æ", "n", "ə"], ["f", "ɔ", "ɹ", "s", "t", "ɹ"]]},
    {"language": "eng", "graphemes": "Anna Foster", "phonemes": "/ænə fɔstɹ/", "segments": [["æ", "n", "ə"], ["f", "ɔ", "s", "t", "ɹ"]]},
    {"language": "eng", "graphemes": "Beth McCarthy-Miller", "phonemes": "/bɛθ məkɑɹði mɪləɹ/", "segments": [["b", "ɛ", "θ"], ["m", "ə", "k", "ɑ", "ɹ", "ð", "i"], ["m", "ɪ", "l", "ə", "ɹ"]]},
    {"language": "eng", "graphemes": "Brett Baer", "phonemes": "/bɹɛt bɛɹ/", "segments": [["b", "ɹ", "ɛ", "t"], ["b", "ɛ", "ɹ"]]},
    {"language": "eng", "graphemes": "Carly Wray", "phonemes": "/kɑɹli ɹeɪ/", "segments": [["k", "ɑ", "ɹ", "l", "i"], ["ɹ", "eɪ"]]},
    {"language": "eng", "graphemes": "Charles Yu", "phonemes": "/t̠ʃɑrlz ju/", "segments": [["t̠ʃ", "ɑ", "r", "l", "z"], ["j", "u"]]},
    {"language": "eng", "graphemes": "Charlie Yu", "phonemes": "/t̠ʃɑrli ju/", "segments": [["t̠ʃ", "ɑ", "r", "l", "i"], ["j", "u"]]},
    {"language": "eng", "graphemes": "Christina Ham", "phonemes": "/kɹəstini  ham/", "segments": [["k", "ɹ", "ə", "s", "t", "ɪ", "n", "ə"], ["h", "æ", "m"]]},
    {"language": "eng", "graphemes": "Claire Cowperthwaite", "phonemes": "/klɛɹ kɑwpɹθweɪt/", "segments": [["k", "l", "ɛ", "ɹ"], ["k", "ɑ", "ʊ", "p", "ɹ", "θ", "w", "eɪ", "t"]]},
    {"language": "eng", "graphemes": "Colleen McGuinness", "phonemes": "/kɑlin məkɪnəs/", "segments": [["k", "ɑ", "l", "ɪ", "n"], ["m", "ə", "k", "ɪ", "n", "ə", "s"]]},
    {"language": "eng", "graphemes": "Colleen McGuinness", "phonemes": "/kɑlin məɡwɪnəs/", "segments": [["k", "ɑ", "l", "ɪ", "n"], ["m", "ə", "ɡ", "w", "ɪ", "n", "ə", "s"]]},
    {"language": "eng", "graphemes": "Constantine Makris", "phonemes": "/kɑnstəntin meɪkɹis/", "segments": [["k", "ɑ", "n", "s", "t", "ə", "n", "t", "ɪ", "n"], ["m", "eɪ", "k", "ɹ", "ɪ", "s"]]},
    {"language": "eng", "graphemes": "Craig William Macneill", "phonemes": "/kɹeɪɡ wɪljəm məknil/", "segments": [["k", "ɹ", "eɪ", "ɡ"], ["w", "ɪ", "l", "j", "ə", "m"], ["m", "ə", "k", "n", "ɪ", "l"]]},
    {"language": "eng", "graphemes": "Craig Zobel", "phonemes": "/kɹeɪɡ zoʊbəl/", "segments": [["k", "ɹ", "eɪ", "ɡ"], ["z", "oʊ", "b", "ə", "l"]]},
    {"language": "eng", "graphemes": "Daisy Gardner", "phonemes": "/deɪzi ɡɑɹdnəɹ/", "segments": [["d", "eɪ", "z", "i"], ["ɡ", "ɑ", "ɹ", "d", "n", "ə", "ɹ"]]},
    {"language": "eng", "graphemes": "Dan Dietz", "phonemes": "/dæn dits/", "segments": [["d", "æ", "n"], ["d", "i", "t", "s"]]},
    {"language": "eng", "graphemes": "Dan Dietz", "phonemes": "/dæn diʌts/", "segments": [["d", "æ", "n"], ["d", "i", "ʌ", "t", "s"]]},
    {"language": "eng", "graphemes": "Daniel T. Thomsen", "phonemes": "/dænjəl ti tɑmsən/", "segments": [["d", "æ", "n", "j", "ə", "l"], ["t", "i"], ["t", "ɑ", "m", "s", "ə", "n"]]},
    {"language": "eng", "graphemes": "Daniel Thomsen", "phonemes": "/dænjəl ti tɑmsən/", "segments": [["d", "æ", "n", "j", "ə", "l"], ["t", "ɑ", "m", "s", "ə", "n"]]},
    {"language": "eng", "graphemes": "Dave Finkel", "phonemes": "/deɪv fɪŋkəl/", "segments": [["d", "eɪ", "v"], ["f", "ɪ", "ŋ", "k", "ə", "l"]]},
    {"language": "eng", "graphemes": "David Finkel", "phonemes": "/deɪvɪd fɪŋkəl/", "segments": [["d", "eɪ", "v", "ɪ", "d"], ["f", "ɪ", "ŋ", "k", "ə", "l"]]},
    {"language": "eng", "graphemes": "Denise Thé", "phonemes": "/dəniz θeɪ/", "segments": [["d", "ə", "n", "ɪ", "z"], ["θ", "eɪ"]]},
    {"language": "eng", "graphemes": "Denise Thé", "phonemes": "/dənis θeɪ/", "segments": [["d", "ə", "n", "ɪ", "s"], ["θ", "eɪ"]]},
    {"language": "eng", "graphemes": "Denise Thé", "phonemes": "/dəniz Teɪ/", "segments": [["d", "ə", "n", "ɪ", "z"], ["t", "eɪ"]]},
    {"language": "eng", "graphemes": "Denise Thé", "phonemes": "/dənis Teɪ/", "segments": [["d", "ə", "n", "ɪ", "s"], ["t", "eɪ"]]},
    {"language": "eng", "graphemes": "Dennie Gordon", "phonemes": "/dɛni gɔɹdən/", "segments": [["d", "ɛ", "n", "i"], ["ɡ", "ɔ", "ɹ", "d", "ə", "n"]]},
    {"language": "eng", "graphemes": "Desa Larkin-Boutté", "phonemes": "/dɛsə lɑɹkɪn buteɪ/", "segments": [["d", "ɛ", "s", "ə"], ["l", "ɑ", "ɹ", "k", "ɪ", "n"], ["b", "u", "t", "eɪ"]]},
    {"language": "eng", "graphemes": "Dominic Mitchell", "phonemes": "/dɑmɪnɪk mɪt̠ʃəl/", "segments": [["d", "ɑ", "m", "ɪ", "n", "ɪ", "k"], ["m", "ɪ", "t̠ʃ", "ə", "l"]]},
    {"language": "eng", "graphemes": "Dominique Mitchell", "phonemes": "/dɑmɪnik mɪt̠ʃəl/", "segments": [["d", "ɑ", "m", "ɪ", "n", "i", "k"], ["m", "ɪ", "t̠ʃ", "ə", "l"]]},
    {"language": "eng", "graphemes": "Don Scardino", "phonemes": "/dɑn skɑɹdinoʊ/", "segments": [["d", "ɑ", "n"], ["s", "k", "ɑ", "ɹ", "d", "ɪ", "n", "oʊ"]]},
    {"language": "eng", "graphemes": "Don Scardino", "phonemes": "/dɑn skɑɹdinoʊ/", "segments": [["d", "ɑ", "n"], ["s", "k", "ɑ", "ɹ", "d", "i", "n", "oʊ"]]},
    {"language": "eng", "graphemes": "Donald Glover", "phonemes": "/dɑnəld ɡlʌvəɹ/", "segments": [["d", "ɑ", "n", "ə", "l", "d"], ["ɡ", "l", "ʌ", "v", "ə", "ɹ"]]},
    {"language": "eng", "graphemes": "Dylan Morgan", "phonemes": "/daɪlən mɔɹɡən/", "segments": [["d", "aɪ", "l", "ə", "n"], ["m", "ɔ", "ɹ", "ɡ", "ə", "n"]]},
    {"language": "eng", "graphemes": "Ed Brubaker", "phonemes": "/ɛd bɹubeɪkəɹ/", "segments": [["ɛ", "d"], ["b", "ɹ", "u", "b", "eɪ", "k", "ə", "ɹ"]]},
    {"language": "eng", "graphemes": "Frederick E.O. Toye", "phonemes": "/fɹɛdəɹɪk i oʊ tɔɪ/", "segments": [["f", "ɹ", "ɛ", "d", "ə", "ɹ", "ɪ", "k"], ["i"], ["oʊ"], ["t", "ɔɪ"]]},
    {"language": "eng", "graphemes": "Gail Mancuso", "phonemes": "/ɡeɪl mæŋkusoʊ/", "segments": [["ɡ", "eɪ", "l"], ["m", "æ", "ŋ", "k", "u", "s", "oʊ"]]},
    {"language": "eng", "graphemes": "Gina Atwater", "phonemes": "/d̠ʒinə ætwɑtər/", "segments": [["d̠ʒ", "i", "n", "ə"], ["æ", "t", "w", "ɑ", "t", "ə", "r"]]},
    {"language": "eng", "graphemes": "Halley Wegryn Gross", "phonemes": "/hæli wɛɡɹin ɡroʊs/", "segments": [["h", "æ", "l", "i"], ["w", "ɛ", "ɡ", "ɹ", "i", "n"], ["ɡ", "r", "oʊ", "s"]]},
    {"language": "eng", "graphemes": "Hanelle M. Culpepper", "phonemes": "/hənɛl ɛm kʌlpəpɚ/", "segments": [["h", "ə", "n", "ɛ", "l"], ["ɛ", "m"], ["k", "ʌ", "l", "p", "ə", "p", "ɚ"]]},
    {"language": "eng", "graphemes": "Hannibal Buress", "phonemes": "/hænəbəl bʌɹɛs/", "segments": [["h", "æ", "n", "ə", "b", "ə", "l"], ["b", "ʌ", "ɹ", "ɛ", "s"]]},
    {"language": "eng", "graphemes": "Helen Shaver", "phonemes": "/hɛlən ʃeɪvər/", "segments": [["h", "ɛ", "l", "ə", "n"], ["ʃ", "eɪ", "v", "ə", "r"]]},
    {"language": "eng", "graphemes": "Jack Burditt", "phonemes": "/d̠ʒæk bɛrdɪt/", "segments": [["d̠ʒ", "æ", "k"], ["b", "ɛ", "r", "d", "ɪ", "t"]]},
    {"language": "eng", "graphemes": "Jamie Sheridan", "phonemes": "/d̠ʒeɪmi ʃɛrɪdən/", "segments": [["d̠ʒ", "eɪ", "m", "i"], ["ʃ", "ɛ", "r", "ɪ", "d", "ə", "n"]]},
    {"language": "eng", "graphemes": "Jeff Richmond", "phonemes": "/d̠ʒɛf ɹɪt̠ʃmənd/", "segments": [["d̠ʒ", "ɛ", "f"], ["ɹ", "ɪ", "t̠ʃ", "m", "ə", "n", "d"]]},
    {"language": "eng", "graphemes": "Jennifer Getzinger", "phonemes": "/d̠ʒɛnɪfɚ ɡɛtsɪnd̠ʒɚ/", "segments": [["d̠ʒ", "ɛ", "n", "ɪ", "f", "ɚ"], ["ɡ", "ɛ", "t", "s", "ɪ", "n", "d̠ʒ", "ɚ"]]},
    {"language": "eng", "graphemes": "John Riggi", "phonemes": "/d̠ʒɑn rɪɡi/", "segments": [["d̠ʒ", "ɑ", "n"], ["r", "ɪ", "ɡ", "i"]]},
    {"language": "eng", "graphemes": "Jon Haller", "phonemes": "/d̠ʒɑn hɔlɚ/", "segments": [["d̠ʒ", "ɑ", "n"], ["h", "ɔ", "l", "ɚ"]]},
    {"language": "eng", "graphemes": "Jon Pollack", "phonemes": "/d̠ʒɑn pɑlək/", "segments": [["d̠ʒ", "ɑ", "n"], ["p", "ɑ", "l", "ə", "k"]]},
    {"language": "eng", "graphemes": "Jonathan Nolan", "phonemes": "/d̠ʒɑnəθən nəʊlən/", "segments": [["d̠ʒ", "ɑ", "n", "ə", "θ", "ə", "n"], ["n", "əʊ", "l", "ə", "n"]]},
    {"language": "eng", "graphemes": "Jonny Campbell", "phonemes": "/d̠ʒɑni kæmbəl/", "segments": [["d̠ʒ", "ɑ", "n", "i"], ["k", "æ", "m", "b", "ə", "l"]]},
    {"language": "eng", "graphemes": "Jordan Goldberg", "phonemes": "/d̠ʒɔɹdən ɡoʊldbɛrɡ/", "segments": [["d̠ʒ", "ɔ", "ɹ", "d", "ə", "n"], ["ɡ", "oʊ", "l", "d", "b", "ɛ", "r", "ɡ"]]},
    {"language": "eng", "graphemes": "Josh Siegal", "phonemes": "/d̠ʒɑʃ siɡəl/", "segments": [["d̠ʒ", "ɑ", "ʃ"], ["s", "i", "ɡ", "ə", "l"]]},
    {"language": "eng", "graphemes": "Juan José Campanella", "phonemes": "/hwɑn hose kæmpənɛlə/", "segments": [["h", "w", "ɑ", "n"], ["h", "oʊ", "s", "e"], ["k", "æ", "m", "p", "ə", "n", "ɛ", "l", "ə"]]},
    {"language": "eng", "graphemes": "Karrie Crouse", "phonemes": "/kæri kɹaʊs/", "segments": [["k", "æ", "r", "i"], ["k", "ɹ", "aʊ", "s"]]},
    {"language": "eng", "graphemes": "Kath Lingenfelter", "phonemes": "/kæθ lɪŋinfɛltəɹ/", "segments": [["k", "æ", "θ"], ["l", "ɪ", "ŋ", "ɪ", "n", "f", "ɛ", "l", "t", "ə", "ɹ"]]},
    {"language": "eng", "graphemes": "Kay Cannon", "phonemes": "/keɪ kænən/", "segments": [["k", "eɪ"], ["k", "æ", "n", "ə", "n"]]},
    {"language": "eng", "graphemes": "Ken Whittingham", "phonemes": "/kɛn wɪtɪŋhæm/", "segments": [["k", "ɛ", "n"], ["w", "ɪ", "t", "ɪ", "ŋ", "h", "æ", "m"]]},
    {"language": "eng", "graphemes": "Kevin Lau", "phonemes": "/kɛvɪn laʊ/", "segments": [["k", "ɛ", "v", "ɪ", "n"], ["l", "aʊ"]]},
    {"language": "eng", "graphemes": "Kevin Rodney Sullivan", "phonemes": "/kɛvɪn rɑdni sʌlvən/", "segments": [["k", "ɛ", "v", "ɪ", "n"], ["r", "ɑ", "d", "n", "i"], ["s", "ʌ", "l", "v", "ə", "n"]]},
    {"language": "eng", "graphemes": "Kevin Rodney Sullivan", "phonemes": "/kɛvɪn rɑdni sʌlvɪn/", "segments": [["k", "ɛ", "v", "ɪ", "n"], ["r", "ɑ", "d", "n", "i"], ["s", "ʌ", "l", "v", "ɪ", "n"]]},
    {"language": "eng", "graphemes": "Lang Fisher", "phonemes": "/læŋ fɪʃəɹ/", "segments": [["l", "æ", "ŋ"], ["f", "ɪ", "ʃ", "ə", "ɹ"]]},
    {"language": "eng", "graphemes": "Lauren Gurganous", "phonemes": "/lɔɹən gɚɡənəs/", "segments": [["l", "ɔ", "ɹ", "ə", "n"], ["ɡ", "ɚ", "ɡ", "ə", "n", "ə", "s"]]},
    {"language": "eng", "graphemes": "Lisa Joy", "phonemes": "/lisa d̠ʒɔɪ/", "segments": [["l", "i", "s", "ə"], ["d̠ʒ", "ɔɪ"]]},
    {"language": "eng", "graphemes": "Matt Pitts", "phonemes": "/mæt pɪts/", "segments": [["m", "æ", "t"], ["p", "ɪ", "t", "s"]]},
    {"language": "eng", "graphemes": "Meera Menon", "phonemes": "/miəɹə mɛnən/", "segments": [["m", "i", "ə", "ɹ", "ə"], ["m", "ɛ", "n", "ə", "n"]]},
    {"language": "eng", "graphemes": "Michael Crichton", "phonemes": "/maɪkəl kraɪtən/", "segments": [["m", "aɪ", "k", "ə", "l"], ["k", "r", "aɪ", "t", "ə", "n"]]},
    {"language": "eng", "graphemes": "Michelle MacLaren", "phonemes": "/mɪʃɛl məklɛɹən/", "segments": [["m", "ɪ", "ʃ", "ɛ", "l"], ["m", "ə", "k", "l", "ɛ", "ɹ", "ə", "n"]]},
    {"language": "eng", "graphemes": "Neil Marshall", "phonemes": "/nil mɑɹʃəl/", "segments": [["n", "i", "l"], ["m", "ɑ", "ɹ", "ʃ", "ə", "l"]]},
    {"language": "eng", "graphemes": "Nicole Kassell", "phonemes": "/nikoʊl kəsɛl/", "segments": [["n", "i", "k", "oʊ", "l"], ["k", "ə", "s", "ɛ", "l"]]},
    {"language": "eng", "graphemes": "Paul Cameron", "phonemes": "/pɔl kæmɹən/", "segments": [["p", "ɔ", "l"], ["k", "æ", "m", "ɹ", "ə", "n"]]},
    {"language": "eng", "graphemes": "Richard J. Lewis", "phonemes": "/ɹɪt̠ʃəɹd d̠ʒ luɪs/", "segments": [["ɹ", "ɪ", "t̠ʃ", "ə", "ɹ", "d"], ["d̠ʒ"], ["l", "u", "ɪ", "s"]]},
    {"language": "eng", "graphemes": "Roberto Patino", "phonemes": "/ɹəbɛɹtoʊ pətinoʊ/", "segments": [["ɹ", "ə", "b", "ɛ", "ɹ", "t", "oʊ"], ["p", "ə", "t", "ɪ", "n", "oʊ"]]},
    {"language": "eng", "graphemes": "Ron Fitzgerald", "phonemes": "/ɹɑn fɪtsd̠ʒəɹəld/", "segments": [["ɹ", "ɑ", "n"], ["f", "ɪ", "t", "s", "d̠ʒ", "ə", "ɹ", "ə", "l", "d"]]},
    {"language": "eng", "graphemes": "Stephen Williams", "phonemes": "/stivən wɪljəmz/", "segments": [["s", "t", "i", "v", "ə", "n"], ["w", "ɪ", "l", "j", "ə", "m", "z"]]},
    {"language": "eng", "graphemes": "Suzanne Wrubel", "phonemes": "/suzæn rubɛl/", "segments": [["s", "u", "z", "æ", "n"], ["r", "u", "b", "ɛ", "l"]]},
    {"language": "eng", "graphemes": "Tarik Saleh", "phonemes": "/tæɹɪk səleɪ/", "segments": [["t", "æ", "ɹ", "ɪ", "k"], ["s", "ə", "l", "eɪ"]]},
    {"language": "eng", "graphemes": "Uta Briesewitz", "phonemes": "/jutə bɹisəwɪtz/", "segments": [["j", "u", "t", "ə"], ["b", "ɹ", "i", "s", "ə", "w", "ɪ", "t", "z"]]},
    {"language": "eng", "graphemes": "Vincenzo Natali", "phonemes": "/vɪnt̠ʃɛnzoʊ nətəli/", "segments": [["v", "ɪ", "n", "t̠ʃ", "ɛ", "n", "z", "oʊ"], ["n", "ə", "t", "ə", "l", "i"]]},
    {"language": "eng", "graphemes": "Wes Humphrey", "phonemes": "/wɛs hʌmfɹi/", "segments": [["w", "ɛ", "s"], ["h", "ʌ", "m", "f", "ɹ", "i"]]},
    {"language": "eng", "graphemes": "Will Soodik", "phonemes": "/wɪl sudɪk/", "segments": [["w", "ɪ", "l"], ["s", "u"], ["d", "ɪ", "k"]]},
    {"language": "spa", "graphemes": "Rafael Nadal", "segments": [["r", "a", "f", "a", "e̞", "l"], ["n", "a", "ð͉", "a", "l", "|"]]},
    
)

for obj in objs:
    phonemes = [word + ['|'] for word in obj['segments']]
    tokens.append(
        Token(**{
            'language': obj['language'],
            'graphemes': obj['graphemes'],
            'phonemes': sum((tuple(p) for p in phonemes), ('|',))
        })
    )

queries = (
#    {'language': 'eng', 'graphemes': 'Vince', 'phonemes': ('|', 'v', 'ɪ', 'n', 's', '|')},
    {'language': 'eng', 'graphemes': 'Rafael Nadal', 'phonemes': ('|', 'ɹ', 'ɑ', 'f', 'aɪ', 'l', '|', 'n', 'ə', 'd', 'ə', 'l', '|')},
)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    parser.add_argument(
        '-b', '--bit-size',
        type=int,
        default=DEFAULT_HASHSIZE,
        help='LSH bit size'
    )
    parser.add_argument(
        '-w', '--window-size',
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help='sliding window size for comparing pairs in the list of pairs sorted by LSH bitwise difference',
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='an integer seed value that will be added to the underlying hash seed',
    )
    parser.add_argument(
        '-q', '--queries',
        type=lambda q: Token(**json.loads(q)),
        nargs='+',
        default=[Token(**q) for q in queries],
        help='query for similar tokens',
    )
    args = parser.parse_args()
    Token.hashsize = args.bit_size
    Token.salt = args.seed
    comparisons = pd.DataFrame(
        compare(
            tokens,
            simhash_bits=SIMHASH_BITS,
            window=args.window_size
        )
    )
    print(
        comparisons.sort_values(
            'simhash difference (in bits)',
            ignore_index=True
        ).to_csv(
            sep='\t'
        )
    )
    #print('queries:', *sorted(args.queries), sep='\n', file=sys.stderr)
    #print(
    #    pd.DataFrame(
    #        search(
    #            args.queries,
    #            tokens,
    #            simhash_bits=SIMHASH_BITS,
    #            window=args.window_size
    #        )
    #    ).sort_values(
    #        'simhash difference (in bits)',
    #        ignore_index=True
    #    ).to_csv(sep='\t')
    #)
