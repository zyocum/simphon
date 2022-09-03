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
usage: simphon.py [-h] [-n N_GRAMS_SIZE] [-b BIT_SIZE] [-w WINDOW_SIZE] [-s SEED]

Script with a CLI to show a simple demonstration of phonemic token matching. The similarity of tokens is computed via simhashing of 2D matrices formed by stacking PHOIBLE phoneme feature vectors
for sequences of phonemes.

options:
  -h, --help            show this help message and exit
  -n N_GRAMS_SIZE, --n-grams-size N_GRAMS_SIZE
                        size n for n-grams (default: 4)
  -b BIT_SIZE, --bit-size BIT_SIZE
                        LSH bit size (default: 256)
  -w WINDOW_SIZE, --window-size WINDOW_SIZE
                        sliding window size for comparing pairs in the list of pairs sorted by LSH bitwise difference (default: 2)
  -s SEED, --seed SEED  an integer seed value that will be added to the underlying hash seed (default: 0)
```

Example output formatted with `tabulate`:

```zsh
(simphon) $ ./simphon.py | tabulate -1s $'\t'
ranking pairs by bitwise similarity: 100%|███████████████████████████████████████████████| 2081/2081 [00:00<00:00, 1450373.32pair/s]
    a                                     b                                       simhash difference (in bits)    similarity score
--  ------------------------------------  ------------------------------------  ------------------------------  ------------------
 0  (eng) 〈Zach〉 /z æ k/                (eng) 〈Zack〉 /z æ k/                                             0               1
 1  (jpn) 〈かつら〉 /k a tsː ɯ d̠ a/      (jpn) 〈カツラ〉 /k a tsː ɯ d̠ a/                                   0               1
 2  (heb) 〈צוֹפִית〉 /ts o f i t/          (heb) 〈צוּפִית〉 /ts u f i t/                                      58               0.975
 3  (eng) 〈Zach〉 /z æ k/                (heb) 〈זך〉 /z a k/                                              68               0.97
 4  (eng) 〈Zack〉 /z æ k/                (heb) 〈זך〉 /z a k/                                              68               0.97
 5  (eng) 〈Tsofit〉 /s oʊ f i t/         (heb) 〈צוּפִית〉 /ts u f i t/                                      77               0.967
 6  (eng) 〈Amit〉 /ɑ m i t/              (heb) 〈עמת〉 /a m ɪ t/                                           79               0.966
 7  (eng) 〈Tsofit〉 /s oʊ f i t/         (heb) 〈צוֹפִית〉 /ts o f i t/                                      79               0.966
 8  (eng) 〈Zachariah〉 /z æ k ə ɹ aɪ ə/  (eng) 〈Zachary〉 /z æ k ə ɹ i/                                   79               0.966
 9  (eng) 〈Jenny〉 /d̠ʒ ɛ n i/            (eng) 〈Johnny〉 /d̠ʒ ɑ n i/                                       80               0.965
10  (heb) 〈עמת〉 /a m ɪ t/               (hin) 〈अमित〉 /aː m ɪ t̪/                                         81               0.965
11  (eng) 〈Jennifer〉 /d̠ʒ ɛ n ə f ɚ/     (eng) 〈Jenny〉 /d̠ʒ ɛ n i/                                        86               0.963
12  (eng) 〈Matt〉 /m æ t/                (eng) 〈Nate〉 /n eɪ t/                                           87               0.962
13  (jpn) 〈ただよし〉 /t a d a j o s i/  (jpn) 〈まさよし〉 /m a s a j o s i/                              88               0.962
14  (eng) 〈Nick〉 /n ɪ k/                (eng) 〈Zach〉 /z æ k/                                            91               0.961
15  (eng) 〈Nick〉 /n ɪ k/                (eng) 〈Zack〉 /z æ k/                                            91               0.961
16  (heb) 〈זְכַרְיָה〉 /z ə̆ x a ʁ j a/       (jpn) 〈カツヤ〉 /k a tsː ɯ j a/                                  91               0.961
17  (eng) 〈Amit〉 /ɑ m i t/              (hin) 〈अमित〉 /aː m ɪ t̪/                                         92               0.96
18  (eng) 〈Catherine〉 /k æ θ ə ɹ ɪ n/   (jpn) 〈カツヤ〉 /k a tsː ɯ j a/                                  93               0.96
19  (eng) 〈Nathan〉 /n eɪ θ ə n/         (hin) 〈आदम〉 /ɑ d̪ ə m/                                           93               0.96
20  (eng) 〈Smith〉 /s m ɪ θ/             (fra) 〈Smith〉 /s m i t/                                         94               0.959
21  (eng) 〈Zachariah〉 /z æ k ə ɹ aɪ ə/  (heb) 〈זְכַרְיָה〉 /z ə̆ x a ʁ j a/                                   94               0.959
22  (eng) 〈Matt〉 /m æ t/                (eng) 〈Nick〉 /n ɪ k/                                            95               0.959
23  (eng) 〈Brett〉 /b ɹ ɛ t/             (eng) 〈Jonathan〉 /d̠ʒ ɑ n ə θ ə n/                               96               0.958
24  (eng) 〈Amit〉 /ɑ m i t/              (fra) 〈Smith〉 /s m i t/                                         97               0.958
...
```
