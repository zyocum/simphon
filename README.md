# simphon
Proof-of-concept for measuring similarity of phoneme sequences using locality sensitive hashing (LSH).

## Setup

Requires Python 3.10+

To install dependencies:

```zsh
$ python3 -m venv simphon
$ source simphon/bin/activate
(simphon) $ pip3 install -U pip
(simphon) $ pip3 install -r requirements.txt
```

To use the virtual environment in the Jupyter notebook, run:

```zsh
(simphon) $ ipython kernel install --user --name=ipa
(simphon) $ jupyter notebook simphon.ipynb
```

Then, choose the kernel with the name of the virtual environment:

![Select the "simphon" kernel](kernel-select.png "kernel selection screenshot")

## Using `simphon.py`

There is a CLI you can run from `simphon.py`:

```zsh
(simphon) $ ./simphon.py -h
usage: simphon.py [-h] [-n N_GRAMS_SIZE] [-b {32,64,128}] [-w WINDOW]

options:
  -h, --help            show this help message and exit
  -n N_GRAMS_SIZE, --n-grams-size N_GRAMS_SIZE
                        size n for n-grams (default: 3)
  -b {32,64,128}, --bits {32,64,128}
                        LSH bit size (default: 128)
  -w WINDOW, --window WINDOW
                        sliding window size for comparing LSH bitwise differences (default: 10)
```

Example output formatted with `tabulate`:

```zsh
(simphon) $ ./simphon.py -n 3 -b 128 -w 10 | tabulate -1s $'\t' -F '0.3f'
ranking pairs by bitwise similarity: 100%|█████████████████████████████████████████████████| 703/703 [00:00<00:00, 1279772.44pair/s]
     a                                    b                                      simhash difference (in bits)    similarity score
---  -----------------------------------  -----------------------------------  ------------------------------  ------------------
  0  (eng) Zach /z æ k/                   (eng) Zak /z æ k/                                                 0               1.000
  1  (eng) Catherine /k æ θ ə r ə n/      (eng) Catherine /k æ θ ə r ɪ n/                                  54               0.789
  2  (eng) Brad /b ɹ æ d/                 (eng) Brett /b ɹ ɛ t/                                            68               0.734
  3  (eng) Carl /k ɑ ɹ l/                 (eng) Carlos /k ɑ ɹ l oʊ s/                                      74               0.711
  4  (eng) Jenny /d̠ʒ ɛ n i/               (eng) Johnny /d̠ʒ ɑ n i/                                          77               0.699
  5  (eng) Matt /m æ t/                   (eng) Mike /m aɪ k/                                              80               0.688
  6  (eng) Alex /æ l ə k s/               (eng) Alexander /æ l ə k z æ n d ɚ/                              84               0.672
  7  (eng) Jenny /d̠ʒ ɛ n i/               (eng) Philip /f ɪ l ɪ p/                                         86               0.664
  8  (eng) Carlos /k ɑ ɹ l oʊ s/          (eng) Charles /t̠ʃ ɑ ɹ l z/                                       88               0.656
  9  (eng) Alex /æ l ə k s/               (eng) Alexi /ə l ɛ k s i/                                        88               0.656
 10  (eng) Jonathan /d̠ʒ ɑ n ə θ ə n/      (eng) Tyler /t aɪ l ə r/                                         88               0.656
...
```
