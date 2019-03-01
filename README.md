# Expressively vulgar: The socio-dynamics of vulgarity and its effects on sentiment analysis in social media

**[Please check this paper for details regarding annotation and modeling](http://aclweb.org/anthology/C18-1248)**

**Citation:**
```
@InProceedings{cachola2018vulgar,
  author    = {Cachola, Isabel  and  Holgate, Eric  and  Preo\c{t}iuc-Pietro, Daniel  and  Li, Junyi Jessy},
  title     = {Expressively vulgar: The socio-dynamics of vulgarity and its effects on sentiment analysis in social media},
  booktitle = {Proceedings of the 27th International Conference on Computational Linguistics},
  year      = {2018},
  pages     = {2927--2938},
  url       = {http://aclweb.org/anthology/C18-1248},
}
```

*Use of the data presented here must abide by the Twitter Terms of Service and Developer Policy*

A bi-LSTM that predicts sentiment values, utilizing vulgarity features.

**The three possible vulgarity features are:**
(1) Masking
(2) Insertion
(3) Concatenations

First, run `clean_data.py` to prepare data set for modeling.
`clean_data.py` automatically uses the path `./data/coling_twitter_data.tsv` to the original data set but if your file path
is different then you can change it using the flag `--data_set`. `clean_data.py` saves the cleaned data to `./data/cleaned_data.tsv`.

**Example Usage:**
`python3 clean_data.py --data_set=./data/coling_twitter_data.tsv`

After cleaning data, run `bilstm.py`

**Required parameters:**
- train=path to training data set
- validation_data=path to validation data set
- initial_embed_weights=path to initial embedding weights
- prefix=prefix to save model

For initial embedding weights, we use [200d CBOW embeddings pre-trained on 50M tweets (Astudillo et al., 2015).](https://www.l2f.inesc-id.pt/~wlin/public/embeddings/)
**Optional parameters:**
- rnndim=<rnn dimension, default=128>
- dropout=<dropout rate, default=0.2>
- maxsentlen=<maximum length of tweets by number of words, default=60>
- num_cat=<number of categories, default=5>
- lr=<learning rate, default=0.001>
- only_testing=<boolean if you only want to load a saved model, default=False>

- concat=<boolean if using concat method, default=False>
- insert=<boolean if using insert method, default=False>
- mask=<boolean if using mask method, default=False>

**Example usage:**
`python3 bilstm.py --train=<path> --test=<path> --prefix=example --concat=True`

**Returns:**
- Saves model as h5 and json files to ./training
- Prints summary of model

If a test set is provided:
- Saves predictions of test set to ./training/predictions
- Prints micro mean absolute error
- Prints macro mean absolute error
- Prints per class mean absolute error
