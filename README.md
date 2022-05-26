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
(simphon) $ ipython kernel install --user --name=simphon
(simphon) $ jupyter notebook simphon.ipynb
```

Then, choose the kernel with the name of the virtual environment:

![Select the "simphon" kernel](kernel-select.png "kernel selection screenshot")

## Using `simphon.py`

There is a CLI you can run from `simphon.py`:

```zsh
(simphon) $ ./simphon.py -h                            
usage: simphon.py [-h] [-n N_GRAMS_SIZE] [-b {32,64,128}] [-w WINDOW]

Script with a CLI to show a simple demonstration of phonemic token matching. The similarity of tokens is computed via simhashing of 2D
matrices formed by stacking PHOIBLE phoneme feature vectors for sequences of phonemes.

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
(simphon) $ ./simphon.py | tabulate -1s $'\t' -F '0.3f'
ranking pairs by bitwise similarity: 100%|█████████████████████████████████████████████████████| 1770/1770 [00:00<00:00, 1705079.94pair/s]
      a                                         b                                      simhash difference (in bits)    similarity score
----  ----------------------------------------  -----------------------------------  ------------------------------  ------------------
   0  (eng) Zach /z æ k/                        (eng) Zack /z æ k/                                                0               1.000
   1  (heb) צוֹפִית /ts o f i t/                  (heb) צוּפִית /ts u f i t/                                        133               0.827
   2  (eng) Jenny /d̠ʒ ɛ n i/                    (eng) Johnny /d̠ʒ ɑ n i/                                         145               0.811
   3  (eng) Sophia /s oʊ f i ə/                 (eng) Tsofit /s oʊ f i t/                                       149               0.806
   4  (eng) Tsofit /s oʊ f i t/                 (heb) צוּפִית /ts u f i t/                                        160               0.792
   5  (eng) Zach /z æ k/                        (heb) זך /z a k/                                                163               0.788
   6  (eng) Zack /z æ k/                        (heb) זך /z a k/                                                163               0.788
   7  (ell) Σοφία /s o f i ɐ/                   (eng) Sophia /s oʊ f i ə/                                       165               0.785
   8  (heb) עמת /a m ɪ t/                       (hin) अमित /aː m ɪ t̪/                                           171               0.777
   9  (eng) Tsofit /s oʊ f i t/                 (heb) צוֹפִית /ts o f i t/                                        171               0.777
  10  (eng) Zachariah /z æ k ə ɹ aɪ ə/          (eng) Zachary /z æ k ə ɹ i/                                     172               0.776
  11  (eng) Sophia /s oʊ f i ə/                 (eng) Sophie /s oʊ f i/                                         179               0.767
  12  (eng) Nate /n eɪ t/                       (eng) Nick /n ɪ k/                                              180               0.766
  13  (eng) Amit /ɑ m i t/                      (hin) अमित /aː m ɪ t̪/                                           181               0.764
  14  (eng) Amit /ɑ m i t/                      (heb) עמת /a m ɪ t/                                             182               0.763
  15  (eng) Sophie /s oʊ f i/                   (eng) Tsofit /s oʊ f i t/                                       182               0.763
  16  (eng) Mike /m aɪ k/                       (eng) Nick /n ɪ k/                                              185               0.759
  17  (eng) Zachariah /z æ k ə ɹ aɪ ə/          (heb) זְכַרְיָה /z ə̆ x a ʁ j a/                                     187               0.757
  18  (eng) Nick /n ɪ k/                        (eng) Zach /z æ k/                                              188               0.755
  19  (eng) Nick /n ɪ k/                        (eng) Zack /z æ k/                                              188               0.755
  20  (eng) Matt /m æ t/                        (eng) Nate /n eɪ t/                                             190               0.753
  21  (eng) Kat /k æ t/                         (heb) זך /z a k/                                                190               0.753
  22  (eng) Katsuya /k æ t s uː j ə/            (rus) Алексей /a lʲ e k sʲ e j/                                 191               0.751
  23  (eng) Katsuya /k æ t s uː j ə/            (eng) Zachariah /z æ k ə ɹ aɪ ə/                                191               0.751
  24  (eng) Zach /z æ k/                        (eng) Zachary /z æ k ə ɹ i/                                     191               0.751
  25  (eng) Zachary /z æ k ə ɹ i/               (eng) Zack /z æ k/                                              191               0.751
...
```
