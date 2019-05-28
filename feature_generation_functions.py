###############
#   MODULES   #
###############


import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
nltk.download("stopwords")

import string
import re

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:20.4f}'.format

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

import numpy as np

from scipy import sparse

import time

import io
import boto3


#################
#   FUNCTIONS   #
#################



# intialize stemmer
ps = PorterStemmer()
ls = LancasterStemmer()

# define stopwords
stops = set(stopwords.words('english'))
stops.add('')

approved_stop_words = {"not", "get", "against", "haven", "haven't","aren't",
                       "aren", "should", "shouldn", "shouldn't", "themselves",
                       "them", "under", "over", 'won', "won't", "wouldn'",
                       "wouldn't"}

stops = stops - approved_stop_words


def clean_text(text, stop_ws=stops, stemmer=ps, str_output=True):
    '''
    This auxiliary function cleans text.

    Methods used for cleaning are:
        (1) transform string of text to list of words,
        (2) cleaned (lowercase, remove punctuation) and remove stop words,
        (3) Porter stemming of cleaned (lowercase, remove punctuation) text,
        (4) Lancaster stemming of cleaned (lowercase, remove punctuation),
        (5) cleaned (lowercase, remove punctuation) without removing stop words.

    Inputs:
        text (string) - A string of text.
        stemming (parameter) - either Porter or Lancaster stemming method
        remove_sw (boolean) - True/False remove stop words

    Outputs:
        Cleaned text per the input parameters.
    '''
    t = text.replace("-", " ").split(" ")
    t = [w.strip(string.punctuation) for w in t]

    if stop_ws:
        t = [w.lower() for w in t if w not in stop_ws]

    if stemmer:
        t = [stemmer.stem(w) for w in t]

    if str_output:
        return ' '.join(t)
    else:
        return t



def make_ngrams(preprocessed, n=2, str_output=True):
    '''
    Covert a list of preprocessed strings into ngrams of length n.
    Should return X ngrams of X words less (n - 1).
    '''
    ngrams_tuples = []

    # ensure that all ngrams are of length n by specifying list position of
    # first item in last ngram
    last_ngram_start = len(preprocessed) - (n - 1)

    # for each string from position i through last ngram start position, create
    # a tuple of length n
    for i in range(last_ngram_start):
        ngrams_tuples.append(tuple(preprocessed[i:i + n]))
    if str_output:
        return [' '.join(ngram) for ngram in ngrams_tuples]
    else:
        return ngrams_tuples



def print_elapsed_time(start, end, m):
    print(f"{m}...Elapsed Time:  {round((end - start)/60,3)} minutes")



def generate_features(df):
    start_time = time.perf_counter()

    df['split'] = df["comment_text"].apply(lambda x: x.split(" "))
    split_time = time.perf_counter()
    print_elapsed_time(start_time, split_time, m="Split comments")

    df['cleaned_w_stopwords_str'] = df["comment_text"].apply(clean_text,args=(None,None,True),)
    df['cleaned_w_stopwords'] = df["comment_text"].apply(clean_text,args=(None,None,False),)
    with_stopwords = time.perf_counter()
    print_elapsed_time(split_time, with_stopwords, m="Cleaned with stopwords")

    df['cleaned_no_stem_str'] = df["comment_text"].apply(clean_text,args=(stops,None, True),)
    df['cleaned_no_stem'] = df["comment_text"].apply(clean_text,args=(stops,None,False),)
    without_stopwords = time.perf_counter()
    print_elapsed_time(with_stopwords, without_stopwords, m="Cleaned without stopwords")

    df['cleaned_porter_str'] = df["comment_text"].apply(clean_text,args=(stops,ps,True),)
    df['cleaned_porter'] = df["comment_text"].apply(clean_text,args=(stops,ps,False),)
    porter_time = time.perf_counter()
    print_elapsed_time(without_stopwords, porter_time, m="Stemmed (Porter)")

    df['cleaned_lancaster_str'] = df["comment_text"].apply(clean_text,args=(stops,ls,True),)
    df['cleaned_lancaster'] = df["comment_text"].apply(clean_text,args=(stops,ls,False),)
    lancaster_time = time.perf_counter()
    print_elapsed_time(porter_time, lancaster_time, m="Stemmed (Lancaster)")

    df['bigrams_unstemmed'] = df["cleaned_no_stem"].apply(make_ngrams,args=(2, True),)
    bigrams_time = time.perf_counter()
    print_elapsed_time(lancaster_time, bigrams_time, m="Created bigrams")
    # df['trigram_porter'] = df["cleaned_porter"].apply(make_ngrams,args=(3, True),)
    # df['fourgram_porter'] = df["cleaned_porter"].apply(make_ngrams,args=(4, True),)
    # df['fivegram_porter'] = df["cleaned_porter"].apply(make_ngrams,args=(5, True),)
    #
    # df['bigram_lancaster'] = df["cleaned_lancaster"].apply(make_ngrams,args=(2, True),)
    # df['trigram_lancaster'] = df["cleaned_lancaster"].apply(make_ngrams,args=(3, True),)
    # df['fourgram_lancaster'] = df["cleaned_lancaster"].apply(make_ngrams,args=(4, True),)
    # df['fivegram_lancaster'] = df["cleaned_lancaster"].apply(make_ngrams,args=(5, True),)

    df['perc_upper'] = df["comment_text"].apply(lambda x: 0 if x == 0 else round((len(re.findall(r'[A-Z]',x)) / len(x)), 3))
    pct_upper_time = time.perf_counter()
    print_elapsed_time(bigrams_time, pct_upper_time, m="Calculated uppercase pct")

    df['num_exclam'] = df["comment_text"].apply(lambda x:(len(re.findall(r'!',x))))
    punctuation_time = time.perf_counter()
    print_elapsed_time(pct_upper_time, punctuation_time, m="Count punctuation")

    df['num_words'] = df["split"].apply(lambda x: len(x))
    wordcount_time = time.perf_counter()
    print_elapsed_time(punctuation_time, wordcount_time, m="Count words")

    calc_stopwords_pct = lambda x, y: 0 if y == 0 else round((x - len(y)) / x, 3)
    df['perc_stopwords'] = df[["num_words", "cleaned_no_stem"]].apply(lambda x: calc_stopwords_pct(*x), axis=1)
    stops_pct_time = time.perf_counter()
    print_elapsed_time(wordcount_time, stops_pct_time, m="Count stopwords pct")

    df['num_upper_words'] = df["split"].apply(lambda x: sum(map(str.isupper, x)) )
    ct_upper_time = time.perf_counter()
    print_elapsed_time(stops_pct_time, ct_upper_time, m="Count uppercase words")

    print()
    print("DONE GENERATING FEATURES")

    return df
