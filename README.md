## ALBERT-Persian: A Lite BERT for Self-supervised Learning of Language Representations for the Persian Language

ALBERT-Persian is the first attempt on ALBERT for the Persian Language. The model was trained based on Google's ALBERT BASE Version 2.0 over various writing styles from numerous subjects (e.g., scientific, novels, news) with more than `3.9M` documents, `73M` sentences, and `1.3B` words, like the way we did for [ParsBERT](https://github.com/hooshvare/parsbert).

## Goals
Objective goals during training are as below (after 140K steps).

``` bash
***** Eval results *****
global_step = 140000
loss = 2.0080082
masked_lm_accuracy = 0.6141017
masked_lm_loss = 1.9963315
sentence_order_accuracy = 0.985
sentence_order_loss = 0.06908702
```

## Introduction

ALBERT-Persian trained on a massive amount of public corpora ([Persian Wikidumps](https://dumps.wikimedia.org/fawiki/), [MirasText](https://github.com/miras-tech/MirasText)) and six other manually crawled text data from a various type of websites ([BigBang Page](https://bigbangpage.com/) `scientific`, [Chetor](https://www.chetor.com/) `lifestyle`, [Eligasht](https://www.eligasht.com/Blog/) `itinerary`,  [Digikala](https://www.digikala.com/mag/) `digital magazine`, [Ted Talks](https://www.ted.com/talks) `general conversational`, Books `novels, storybooks, short stories from old to the contemporary era`).

## Results

The following tables summarize the F1 scores obtained by ALBERT-Persian as compared to other models and architectures.


### Sentiment Analysis (SA) Task

|          Dataset         | ALBERT-fa-base-v2 | ParsBERT-v1 | mBERT | DeepSentiPers |
|:------------------------:|:-----------------:|:-----------:|:-----:|:-------------:|
|  Digikala User Comments  |       81.12       |    81.74    | 80.74 |       -       |
|  SnappFood User Comments |       85.79       |    88.12    | 87.87 |       -       |
|  SentiPers (Multi Class) |       66.12       |    71.11    |   -   |     69.33     |
| SentiPers (Binary Class) |       91.09       |    92.13    |   -   |     91.98     |


### Text Classification (TC) Task

|      Dataset      | ALBERT-fa-base-v2 | ParsBERT-v1 | mBERT |
|:-----------------:|:-----------------:|:-----------:|:-----:|
| Digikala Magazine |       92.33       |    93.59    | 90.72 |
|    Persian News   |       97.01       |    97.19    | 95.79 |


### Named Entity Recognition (NER) Task

| Dataset | ALBERT-fa-base-v2 | ParsBERT-v1 | mBERT | MorphoBERT | Beheshti-NER | LSTM-CRF | Rule-Based CRF | BiLSTM-CRF |
|:-------:|:-----------------:|:-----------:|:-----:|:----------:|:------------:|:--------:|:--------------:|:----------:|
|  PEYMA  |       88.99       |    93.10    | 86.64 |      -     |     90.59    |     -    |      84.00     |      -     |
|  ARMAN  |       97.43       |    98.79    | 95.89 |    89.9    |     84.03    |   86.55  |        -       |    77.45   |


**If you tested ALBERT-Persian on a public dataset and you want to add your results to the table above, open a pull request or contact us. Also make sure to have your code available online so we can add it as a reference**


## How to use

### TensorFlow 2.0

```python
from transformers import AutoConfig, AutoTokenizer, TFAutoModel

config = AutoConfig.from_pretrained("m3hrdadfi/albert-fa-base-v2")
tokenizer = AutoTokenizer.from_pretrained("m3hrdadfi/albert-fa-base-v2")
model = TFAutoModel.from_pretrained("m3hrdadfi/albert-fa-base-v2")

text = "ما در هوشواره معتقدیم با انتقال صحیح دانش و آگاهی، همه افراد می‌توانند از ابزارهای هوشمند استفاده کنند. شعار ما هوش مصنوعی برای همه است."
tokenizer.tokenize(text)

>>> ['▁ما', '▁در', '▁هوش', 'واره', '▁معتقد', 'یم', '▁با', '▁انتقال', '▁صحیح', '▁دانش', '▁و', '▁اگاه', 'ی', '،', '▁همه', '▁افراد', '▁می', '▁توانند', '▁از', '▁ابزارهای', '▁هوشمند', '▁استفاده', '▁کنند', '.', '▁شعار', '▁ما', '▁هوش', '▁مصنوعی', '▁برای', '▁همه', '▁است', '.']

```

### Pytorch

```python
from transformers import AutoConfig, AutoTokenizer, AutoModel

config = AutoConfig.from_pretrained("m3hrdadfi/albert-fa-base-v2")
tokenizer = AutoTokenizer.from_pretrained("m3hrdadfi/albert-fa-base-v2")
model = AutoModel.from_pretrained("m3hrdadfi/albert-fa-base-v2")
```

## NLP Tasks Tutorial 

- Named Entity Recognition (soon).
- Sentiment (soon).
- Text Classification (soon).
- Topic modeling (soon).
- Question Answering (soon).


## Cite 

I didn't publish any paper about this work, yet! Please cite this repository in publications as the following:

```markdown
@misc{ALBERT-Persian,
  author = {Mehrdad Farahani},
  title = {ALBERT-Persian: A Lite BERT for Self-supervised Learning of Language Representations for the Persian Language},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/m3hrdadfi/albert-persian}},
}
@article{ParsBERT,
    title={ParsBERT: Transformer-based Model for Persian Language Understanding},
    author={Mehrdad Farahani, Mohammad Gharachorloo, Marzieh Farahani, Mohammad Manthouri},
    journal={ArXiv},
    year={2020},
    volume={abs/2005.12515}
}
```