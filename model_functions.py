###############
#   MODULES   #
###############

import pandas as pd
import string
import re
import string
import numpy as np
import datetime

import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


#################
#   FUNCTIONS   #
#################


def run_model(train_perc, target, see_inside, comments, model_df, model_type):
    '''
    This function runs a single machine learning model as per the specified
    parameters.

    Input(s):
        model_df   - (data frame) source data frame
        train_perc - (float) percentage that should be used for training set
        model_type - (string) which machine learning model to use
        see_inside - (boolean) returns the intermediate tokenized and vectorized
                        arrays
        comments   - (string) source column for text data
        target     - (string) source column for y values

    Output(s):

        clf               - (sklearn object) the classifier model
        output            - (data frame) Predicted Y values for the test set
        X_all_counts      - (array) TF-IDF weights
        X_all_tfidf       - (data frame) Prepared TF-IDF values to on which
                                to run the model
        fitted_vectorizer - (array) Matrix of TF-IDF features

    Citation:  https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
    '''

    train_start = 0
    train_end = round(model_df.shape[0]*train_perc)

    test_start = train_end
    test_end = model_df.shape[0]

    X_all = model_df[comments].values
    y_all = model_df[target].values

    # calculating frequencies
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    fitted_vectorizer=tfidf_vectorizer.fit(model_df[comments].values.astype('U'))
    X_all_tfidf =  fitted_vectorizer.transform(model_df[comments].values.astype('U'))


    X_train = X_all_tfidf[train_start:train_end]
    y_train = model_df[train_start:train_end][target].values
    y_train=y_train.astype('int')


    X_test = X_all_tfidf[test_start:test_end]
    y_test = model_df[test_start:test_end][target].values
    print("fitting model now")
    model_dict = {}
    model_dict["MultiNB"] = MultinomialNB()
    model_dict["GaussNB"] = GaussianNB()
    model_dict['SVM'] = svm.SVC(kernel='linear', probability=True,
                                random_state=1008)
    model_dict["LR"] = LogisticRegression(penalty="l1",C=1e5)

    clf = model_dict[model_type].fit(X_train, y_train)

    predicted = clf.predict(X_test)

    output = model_df[test_start:test_end]
    output['predicted'] = predicted
    output['y_test'] = y_test
    output['accuracy'] = output.predicted == output.y_test

    if see_inside == True:
        return clf, output, X_all_counts, X_all_tfidf
    else:
        return clf, output, fitted_vectorizer



def get_metrics(should_print, detailed, output, round_to):
    '''
    This function returns the model's metrics for various subsets of data.

    Input(s):
        output       - (data frame) Predicted Y values for the test set
        should_print - (boolean) Print out results
        detailed     - (boolean) Whether it should include metrics for identity,
                            obscenity, threats, insults
        round_to     - (integer) number of decimals to round to

    Output(s):
        metrics - (data frame) metric results i.e. accuracy, precision, recall,
                        f1_score, AUC-ROC

    '''
    round_to = 3
    metrics = {}
    targets = output[output.y_test == 1]
    nontargets = output[output.y_test == 0]

    dfs = [output, targets, nontargets]
    labels = ["Overall", "Target", "Non-Target"]

    for i in range(len(dfs)):
        df, label = dfs[i], labels[i]
        if label == "Non-Target":
            pos_label = 0
        else:
            pos_label = 1

        metrics[label] = {}


        accuracy = accuracy_score(df.y_test, df.predicted)
        metrics[label]['Accuracy'] = accuracy

        precision = precision_score(df.y_test, df.predicted, pos_label=pos_label)
        metrics[label]['Precision'] = precision
        recall = recall_score(df.y_test, df.predicted, pos_label=pos_label)
        metrics[label]['Recall'] = recall

        f1 = f1_score(df.y_test, df.predicted, pos_label=pos_label)
        metrics[label]['F1'] = f1
        if label == "Overall":
            roc_auc = round(roc_auc_score(df.y_test, df.predicted), round_to)
            metrics[label]['ROC_AUC'] = roc_auc

        if should_print == True:
            print("{} Accuracy: {}".format(label, accuracy))
            print("{} Precision: {}".format(label, precision))
            print("{} Recall: {}".format(label, recall))
            print("{} F1 Score: {}".format(label, f1))
            if label == "Overall":
                print("ROC_AUC: {}".format(roc_auc))
            print()

    if detailed == True:

        identities = output[output.identity_attack > .5]
        obscenity = output[output.obscene > .5]
        insults = output[output.insult > .5]
        threats = output[output.threat > .5]
        detail_dfs = [identities, obscenity, insults, threats]
        detail_labels = ["Strong Identity", "Obscenity", "Insults", "Threats"]

        for i in range(len(detail_dfs)):
            dfd, labeld = detail_dfs[i], detail_labels[i]
            metrics[label] = {}

            accuracy = accuracy_score(dfd.y_test, dfd.predicted)
            metrics[label]['Accuracy'] = accuracy

            precision = precision_score(dfd.y_test, dfd.predicted)
            metrics[label]['Precision'] = precision
            recall = recall_score(dfd.y_test, dfd.predicted)
            metrics[label]['Recall'] = recall

            f1 = round(f1_score(dfd.y_test, dfd.predicted))
            metrics[label]['F1'] = f1

            if should_print == True:
                print("{} Accuracy: {}".format(labeld, accuracy))
                print("{} Precision: {}".format(labeld, precision))
                print("{} Recall: {}".format(labeld, recall))
                print("{} F1 Score: {}".format(labeld, f1))
            print()

    return metrics



def run_model_test(comments, target, model_df, clf, vectorizer):
    '''

    Input(s):
        model_df - (data frame) the hold out set on which to test the model
        clf - (sklearn object) the classifier model
        vectorizer - (array) Matrix of TF-IDF features
        comments - (string) the column name of the independent variable X
        target - (string) the column name of the dependent variable Y

    Output(s):
        output - (data frame) model_df with added columns for values of
                    true Y ('y_test') and Y-hat ('predicted')
    '''

    # calculating frequencies
    X_all_tfidf = vectorizer.transform(model_df[comments].astype('U'))

    predicted = clf.predict(X_all_tfidf)

    output = model_df
    output['predicted'] = predicted
    output['y_test'] = model_df[target]
    print(output.columns)

    return output
