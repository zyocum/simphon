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
$ ./simphon.py -h
usage: simphon.py [-h] [-b BIT_SIZE] [-w WINDOW_SIZE] [-s SEED] [-q QUERIES [QUERIES ...]]

Script with a CLI to show a simple demonstration of phonemic token matching. The similarity of tokens is computed via simhashing of 2D matrices formed by stacking PHOIBLE phoneme feature vectors for sequences of phonemes.

options:
  -h, --help            show this help message and exit
  -b BIT_SIZE, --bit-size BIT_SIZE
                        LSH bit size (default: 256)
  -w WINDOW_SIZE, --window-size WINDOW_SIZE
                        sliding window size for comparing pairs in the list of pairs sorted by LSH bitwise difference (default: 2)
  -s SEED, --seed SEED  an integer seed value that will be added to the underlying hash seed (default: 0)
  -q QUERIES [QUERIES ...], --queries QUERIES [QUERIES ...]
                        query for similar tokens (default: [Token(language='eng', graphemes='Rafael Nadal', phonemes=('|', 'ɹ', 'ɑ', 'f', 'aɪ', 'l', '|', 'n', 'ə', 'd', 'ə', 'l', '|'))])
```

Example output formatted with `tabulate`:

```zsh
(simphon) $ ./simphon.py | tabulate -1s $'\t'                                                                                    
comparing pairs: 12088pair [02:44, 73.68pair/s]  
       a                                                                            b                                                                              simhash difference (in bits)    similarity score
-----  ---------------------------------------------------------------------------  ---------------------------------------------------------------------------  ------------------------------  ------------------
    0  (jpn) 〈かつら〉 /| k a tsː ɯ d̠ a |/                                         (jpn) 〈カツラ〉 /| k a tsː ɯ d̠ a |/                                                                      0               1
    1  (eng) 〈Zach〉 /| z æ k |/                                                   (eng) 〈Zack〉 /| z æ k |/                                                                                0               1
    2  (eng) 〈Kevin Rodney Sullivan〉 /| k ɛ v ɪ n | r ɑ d n i | s ʌ l v ə n |/    (eng) 〈Kevin Rodney Sullivan〉 /| k ɛ v ɪ n | r ɑ d n i | s ʌ l v ɪ n |/                               287               0.907
    3  (eng) 〈Don Scardino〉 /| d ɑ n | s k ɑ ɹ d i n oʊ |/                        (eng) 〈Don Scardino〉 /| d ɑ n | s k ɑ ɹ d ɪ n oʊ |/                                                   322               0.895
    4  (eng) 〈Adam Bernstein〉 /| æ d ə m | b ɛ r n s t aɪ n |/                    (eng) 〈Adam Bernstein〉 /| æ d ə m | b ɛ r n s t i n |/                                                328               0.893
    5  (eng) 〈Dominic Mitchell〉 /| d ɑ m ɪ n ɪ k | m ɪ t̠ʃ ə l |/                  (eng) 〈Dominique Mitchell〉 /| d ɑ m ɪ n i k | m ɪ t̠ʃ ə l |/                                           334               0.891
    6  (eng) 〈Denise Thé〉 /| d ə n ɪ s | t eɪ |/                                  (eng) 〈Denise Thé〉 /| d ə n ɪ z | t eɪ |/                                                             348               0.887
    7  (eng) 〈Denise Thé〉 /| d ə n ɪ s | θ eɪ |/                                  (eng) 〈Denise Thé〉 /| d ə n ɪ z | θ eɪ |/                                                             397               0.871
    8  (eng) 〈Denise Thé〉 /| d ə n ɪ z | t eɪ |/                                  (eng) 〈Denise Thé〉 /| d ə n ɪ z | θ eɪ |/                                                             431               0.86
    9  (eng) 〈Daniel T. Thomsen〉 /| d æ n j ə l | t i | t ɑ m s ə n |/            (eng) 〈Daniel Thomsen〉 /| d æ n j ə l | t ɑ m s ə n |/                                                432               0.859
   10  (eng) 〈Denise Thé〉 /| d ə n ɪ s | t eɪ |/                                  (eng) 〈Denise Thé〉 /| d ə n ɪ s | θ eɪ |/                                                             452               0.853
   11  (heb) 〈צוֹפִית〉 /| ts o f i t |/                                             (heb) 〈צוּפִית〉 /| ts u f i t |/                                                                        460               0.85
   12  (eng) 〈Jenny〉 /| d̠ʒ ɛ n i |/                                               (eng) 〈Johnny〉 /| d̠ʒ ɑ n i |/                                                                         500               0.837
   13  (eng) 〈Colleen McGuinness〉 /| k ɑ l ɪ n | m ə k ɪ n ə s |/                 (eng) 〈Colleen McGuinness〉 /| k ɑ l ɪ n | m ə ɡ w ɪ n ə s |/                                          504               0.836
   14  (eng) 〈Anna Foerster〉 /| æ n ə | f ɔ ɹ s t ɹ |/                            (eng) 〈Anna Foster〉 /| æ n ə | f ɔ s t ɹ |/                                                           528               0.828
   15  (eng) 〈Dave Finkel〉 /| d eɪ v | f ɪ ŋ k ə l |/                             (eng) 〈David Finkel〉 /| d eɪ v ɪ d | f ɪ ŋ k ə l |/                                                   540               0.824
   16  (eng) 〈Zachariah〉 /| z æ k ə ɹ aɪ ə |/                                     (eng) 〈Zachary〉 /| z æ k ə ɹ i |/                                                                     557               0.819
   17  (eng) 〈Amit〉 /| ɑ m i t |/                                                 (heb) 〈עמת〉 /| a m ɪ t |/                                                                             558               0.818
   18  (eng) 〈Zach〉 /| z æ k |/                                                   (heb) 〈זך〉 /| z a k |/                                                                                559               0.818
   19  (eng) 〈Zack〉 /| z æ k |/                                                   (heb) 〈זך〉 /| z a k |/                                                                                559               0.818
   20  (eng) 〈Andrew Seklir〉 /| æ n d ɹ u | s ɛ k l ə ɹ |/                        (eng) 〈Andrew Seklir〉 /| æ n d ɹ u | s ɛ k l ɪ ɹ |/                                                   565               0.816
   21  (eng) 〈Brett〉 /| b ɹ ɛ t |/                                                (eng) 〈Brett Baer〉 /| b ɹ ɛ t | b ɛ ɹ |/                                                              567               0.815
   22  (eng) 〈Adam Bernstein〉 /| æ d ə m | b ɛ r n s t i n |/                     (eng) 〈Bernstein, Adam〉 /| b ɛ r n s t i n | æ d ə m |/                                               568               0.815
   23  (eng) 〈Sophia〉 /| s oʊ f i ə |/                                            (eng) 〈Tsofit〉 /| s oʊ f i t |/                                                                       580               0.811
   24  (jpn) 〈ただよし〉 /| t a d a j o s i |/                                     (jpn) 〈まさよし〉 /| m a s a j o s i |/                                                                580               0.811
   25  (heb) 〈עמת〉 /| a m ɪ t |/                                                  (hin) 〈अमित〉 /| aː m ɪ t̪ |/                                                                           581               0.811
...
```
