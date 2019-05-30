###############
#   MODULES   #
###############

import string
import nltk
import pandas as pd
# import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
#
nltk.download("stopwords")
#
# import string
import re
#
# import warnings
# warnings.filterwarnings('ignore')
# import matplotlib.pyplot as plt
# pd.options.display.float_format = '{:20.4f}'.format
#
# import sklearn
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
#
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.naive_bayes import GaussianNB
#
# from sklearn.linear_model import SGDClassifier
#
import numpy as np
#
# from scipy import sparse
#
import time



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
    This auxiliary function cleans text.  Fed to the generate_features function.

    Methods used for cleaning are:
        (1) transform string of text to list of words,
        (2) cleaned (lowercase, remove punctuation) and remove stop words,
        (3) Porter stemming of cleaned (lowercase, remove punctuation) text,
        (4) Lancaster stemming of cleaned (lowercase, remove punctuation),
        (5) cleaned (lowercase, remove punctuation) without removing stop words.

    Inputs:
        text (string)        - A string of text.
        stemming (parameter) - either Porter or Lancaster stemming method
        remove_sw (boolean)  - True/False remove stop words

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
    This auxiliary function converts a list of preprocessed strings into
    ngrams of length n.  Returns X ngrams of X words less (n - 1).
    Fed to the generate_features function.

    Input(s):
        preprocessed - (list) List of preprocessed strings
        n            - (int) Length of n-gram
        str_output   - (boolean) output as a string (true) or list (false)

    Output(s):
        Either a list or string of ngrams

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
    '''
    This auxiliary function prints elapsed time.  Fed to the generate_features function.
    '''
    print(f"{m}...Elapsed Time:  {round((end - start)/60,3)} minutes")



def generate_all_features(df):
    '''
    This function generates text and numerical features for models.

    Input(s):
        df - (data frame) raw data

    Output(s):
        df - (data frame) returns the data frame with added features:
            (1) split - (list) Comment string split into a list of words
            (2) cleaned_w_stopwords_str - (string) Comment with punctuation removed
            (3) cleaned_w_stopwords - (list) Comment with punctuation removed, split into list of words
            (4) cleaned_no_stem_str - (string) Comment with stopwords and punctuation removed, lowercased
            (5) cleaned_no_stem - (list) Comment with stopwords and punctuation removed, lowercased,
                                   and split into list of words
            (6) cleaned_porter_str - (string) Comment with stopwords and punctuation removed, lowercased,
                                      and Porter stemmed
            (7) cleaned_porter - (list) Comment with stopwords and punctuation removed, lowercased,
                                  Porter stemmed, and split into list of words
            (8) cleaned_lancaster_str - (string) Comment with stopwords and punctuation removed, lowercased,
                                         and Lancaster stemmed
            (9) cleaned_lancaster - (list) Comment with stopwords and punctuation removed, lowercased,
                                    Lancaster stemmed, and split into list of words
            (10) bigrams_unstemmed - Comment with stopwords and punctuation removed, lowercased,
                                     then converted into bigrams
            (11) perc_upper - Percent of uppercase letters in comment
            (12) num_exclam - Count of exclamation points in comment
            (13) num_words - Count of words in comment

    '''
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




def generate_NB_SVM_features(df):
    '''
    This function generates text and numerical features for models.

    Input(s):
        df - (data frame) raw data

    Output(s):
        df - (data frame) returns the data frame with added features:
            (1) split - (list) Comment string split into a list of words
            (2) cleaned_w_stopwords_str - (string) Comment with punctuation removed
            (3) cleaned_no_stem_str - (string) Comment with stopwords and punctuation removed, lowercased
            (4) cleaned_porter_str - (string) Comment with stopwords and punctuation removed, lowercased,
                                      and Porter stemmed
            (5) cleaned_lancaster_str - (string) Comment with stopwords and punctuation removed, lowercased,
                                         and Lancaster stemmed
            
    '''
    start_time = time.perf_counter()

    df['cleaned_w_stopwords_str'] = df["comment_text"].apply(clean_text,args=(None,None,True),)
    with_stopwords = time.perf_counter()
    print_elapsed_time(start_time, with_stopwords, m="Cleaned with stopwords")

    df['cleaned_no_stem_str'] = df["comment_text"].apply(clean_text,args=(stops,None, True),)
    without_stopwords = time.perf_counter()
    print_elapsed_time(with_stopwords, without_stopwords, m="Cleaned without stopwords")

    df['cleaned_porter_str'] = df["comment_text"].apply(clean_text,args=(stops,ps,True),)
    porter_time = time.perf_counter()
    print_elapsed_time(without_stopwords, porter_time, m="Stemmed (Porter)")

    df['cleaned_lancaster_str'] = df["comment_text"].apply(clean_text,args=(stops,ls,True),)
    lancaster_time = time.perf_counter()
    print_elapsed_time(porter_time, lancaster_time, m="Stemmed (Lancaster)")

    print()
    print("DONE GENERATING FEATURES")

    return df
