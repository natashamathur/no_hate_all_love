###############
#   MODULES   #
###############

# import pandas as pd
# import string
# import re
# import string
# import numpy as np
# import datetime
#
# import warnings
# warnings.filterwarnings('ignore')
#
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.stem import LancasterStemmer
#
# import sklearn
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# from sklearn.naive_bayes import MultinomialNB,GaussianNB
# from sklearn import svm
# from sklearn.linear_model import LogisticRegression
#
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


#################
#   FUNCTIONS   #
#################


def run_model(model_df, train_perc=.80,  model_type = "SVM",
             see_inside=False, comments="comment_text",
             target='toxicity_category'):
    '''
    This function runs a single machine learning model as per the specified parameters.

    Input(s):
        model_df: source data frame
        train_perc: percentage that should be used for training set
        addtl_feats: (list) list of non text columns to include
        model_type: which machine learning model to use
        see_inside: returns the intermediate tokenized and vectorized arrays
        comments: source column for text data
        target: source column for y values

    Output(s):

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
    model_dict['SVM'] = svm.SVC(kernel='linear', probability=True, random_state=1008)
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
        return clf, output






def get_metrics(output, should_print=True, round_to=3):
    '''
    This function returns the model's metrics.

    '''
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


        accuracy = round(accuracy_score(df.y_test, df.predicted), round_to)
        metrics[label]['Accuracy'] = accuracy

        precision = round(precision_score(df.y_test, df.predicted, pos_label=pos_label), round_to)
        metrics[label]['Precision'] = precision

        recall = round(recall_score(df.y_test, df.predicted, pos_label=pos_label), round_to)
        metrics[label]['Recall'] = recall

        f1 = round(f1_score(df.y_test, df.predicted, pos_label=pos_label), round_to)
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
                print("ROC_AUC: {}".format(label, roc_auc))
            print()

    return metrics
