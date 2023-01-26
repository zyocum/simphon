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
usage: simphon.py [-h] [-b BIT_SIZE] [-w WINDOW_SIZE] [-s SEED]

Script with a CLI to show a simple demonstration of phonemic token matching. The similarity of tokens is computed via simhashing
of 2D matrices formed by stacking PHOIBLE phoneme feature vectors for sequences of phonemes.

options:
  -h, --help            show this help message and exit
  -b BIT_SIZE, --bit-size BIT_SIZE
                        LSH bit size (default: 256)
  -w WINDOW_SIZE, --window-size WINDOW_SIZE
                        sliding window size for comparing pairs in the list of pairs sorted by LSH bitwise difference (default: 8)
  -s SEED, --seed SEED  an integer seed value that will be added to the underlying hash seed (default: 0)
```

Example output formatted with `tabulate`:

```zsh
(simphon) $ ./simphon.py | tabulate -1s $'\t'
ranking pairs by bitwise similarity: 2145pair [00:42, 50.03pair/s] 
      a                                             b                                          simhash difference (in bits)    similarity score
----  --------------------------------------------  ---------------------------------------  ------------------------------  ------------------
   0  (jpn) 〈かつら〉 /k a tsː ɯ d̠ a/              (jpn) 〈カツラ〉 /k a tsː ɯ d̠ a/                                      0               1
   1  (eng) 〈Zach〉 /z æ k/                        (eng) 〈Zack〉 /z æ k/                                                0               1
   2  (heb) 〈צוֹפִית〉 /ts o f i t/                  (heb) 〈צוּפִית〉 /ts u f i t/                                        470               0.847
   3  (eng) 〈Amit〉 /ɑ m i t/                      (heb) 〈עמת〉 /a m ɪ t/                                             560               0.818
   4  (jpn) 〈ただよし〉 /t a d a j o s i/          (jpn) 〈まさよし〉 /m a s a j o s i/                                571               0.814
   5  (eng) 〈Zack〉 /z æ k/                        (heb) 〈זך〉 /z a k/                                                586               0.809
   6  (eng) 〈Zach〉 /z æ k/                        (heb) 〈זך〉 /z a k/                                                586               0.809
   7  (eng) 〈Zachariah〉 /z æ k ə ɹ aɪ ə/          (eng) 〈Zachary〉 /z æ k ə ɹ i/                                     598               0.805
   8  (heb) 〈עמת〉 /a m ɪ t/                       (hin) 〈अमित〉 /aː m ɪ t̪/                                           602               0.804
   9  (eng) 〈Tsofit〉 /s oʊ f i t/                 (heb) 〈צוּפִית〉 /ts u f i t/                                        614               0.8
  10  (eng) 〈Amit〉 /ɑ m i t/                      (hin) 〈अमित〉 /aː m ɪ t̪/                                           628               0.796
  11  (eng) 〈Sophia〉 /s oʊ f i ə/                 (eng) 〈Tsofit〉 /s oʊ f i t/                                       641               0.791
  12  (eng) 〈Sophie〉 /s oʊ f i/                   (eng) 〈Tsofit〉 /s oʊ f i t/                                       641               0.791
  13  (eng) 〈Tsofit〉 /s oʊ f i t/                 (heb) 〈צוֹפִית〉 /ts o f i t/                                        656               0.786
  14  (eng) 〈Alexander〉 /æ l ɪ k z æ n d ɚ/       (eng) 〈Xander〉 /z æ n d ɚ/                                        660               0.785
  15  (ell) 〈Σοφία〉 /s o f i ɐ/                   (eng) 〈Sophia〉 /s oʊ f i ə/                                       669               0.782
  16  (eng) 〈Sophia〉 /s oʊ f i ə/                 (eng) 〈Sophie〉 /s oʊ f i/                                         672               0.781
  17  (eng) 〈Carl〉 /k ɑ ɹ l/                      (eng) 〈Carlos〉 /k ɑ ɹ l oʊ s/                                     677               0.78
  18  (eng) 〈Brad〉 /b ɹ æ d/                      (eng) 〈Bradley〉 /b ɹ æ d l i/                                     688               0.776
  19  (eng) 〈Mike〉 /m aɪ k/                       (eng) 〈Nate〉 /n eɪ t/                                             700               0.772
  20  (eng) 〈Zachary〉 /z æ k ə ɹ i/               (jpn) 〈カツヤ〉 /k a tsː ɯ j a/                                    704               0.771
  21  (eng) 〈Mike〉 /m aɪ k/                       (eng) 〈Nick〉 /n ɪ k/                                              707               0.77
  22  (eng) 〈Phil〉 /f ɪ l/                        (eng) 〈Philip〉 /f ɪ l ɪ p/                                        721               0.765
  23  (eng) 〈Matt〉 /m æ t/                        (eng) 〈Mike〉 /m aɪ k/                                             721               0.765
  24  (eng) 〈Kat〉 /k æ t/                         (heb) 〈זך〉 /z a k/                                                724               0.764
...
```
