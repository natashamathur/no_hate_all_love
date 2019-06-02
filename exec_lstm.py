#!/usr/bin/python3.7

import os
import re
import sys
import ssl
import json
import nltk
import time
import math
import string
import pickle
import argparse

import torch
import numpy as np
import pandas as pd
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence, pack_padded_sequence


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

TOXIC_LABEL = 'toxic'
NOT_TOXIC_LABEL = 'not_toxic'
VOCAB_SIZE = 7500

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using CUDA? {USE_CUDA} {device}")



def load_label(filepath):
    df = pd.read_csv(filepath)
    df['label'] = df.target.apply(lambda x: 1 if x > 0.5 else 0)
    print(f"{filepath} loaded successfully!")
    return df


def portion_data(df, ratio):
    msk = np.random.rand(len(df)) <= ratio
    set1_df = df[msk]
    set2_df = df[~msk]
    return set1_df, set2_df


def get_samples(df, proportion=0.2, train_test_ratio=0.8):
    print(f"Retrieving training samples at proportion {proportion}")
    train_set, val_set = portion_data(df, train_test_ratio)
    train_sample = train_set.sample(frac=proportion, replace=True, random_state=1008)

    val_set, test_set = portion_data(val_set, ratio=0.5)

    return train_sample, val_set, test_set



def rebalance_data(train_sample, rebalance_ratios=[0.35, 0.5, 0.6, 0.65, 0.75]):
    toxic = train_sample[train_sample.label == 1]
    nontoxic = train_sample[train_sample.label == 0]

    TOTAL_TOXIC = len(toxic)
    TOTAL_SAMPLES = train_sample.shape[0]
    all_rebalanced = []

    for toxic_ratio in rebalance_ratios:
        # check how many times need to repeat total toxic to get test_ratio
        DESIRED_TOXIC = int(TOTAL_SAMPLES * toxic_ratio)
        print(f"Rebalance Ratio: {toxic_ratio}, {DESIRED_TOXIC} toxic samples out of {TOTAL_SAMPLES}")
        DESIRED_NONTOXIC = int(TOTAL_SAMPLES - DESIRED_TOXIC)
        required_repeats = DESIRED_TOXIC // TOTAL_TOXIC
        # rebalanced_df = pd.DataFrame(np.repeat(toxic.values,required_repeats,axis=0))
        rebalanced_df = toxic.iloc[np.repeat(np.arange(len(toxic)), required_repeats)]

        rebalanced_df = rebalanced_df.append(nontoxic.sample(n=DESIRED_NONTOXIC, random_state=1008))
        rebalanced_df = rebalanced_df.sample(frac=1, random_state=1008).reset_index(drop=True)
        # print(rebalanced_df.head())
        all_rebalanced.append(rebalanced_df)

    # prepared_35 = toxic.append(nontoxic.sample(TOTAL_TOXIC*3))
    # prepared_35 = prepared_35.sample(frac=1).reset_index(drop=True)
    #
    # prepared_50 = toxic.append(toxic).append(nontoxic.sample(TOTAL_TOXIC*2))
    # prepared_50 = prepared_50.sample(frac=1).reset_index(drop=True)
    #
    # prepared_60 = toxic.append(toxic).append(toxic).append(nontoxic.sample(TOTAL_TOXIC))
    # prepared_60 = prepared_60.sample(frac=1).reset_index(drop=True)
    #
    # prepared_65 = toxic.append(toxic).append(toxic).append(nontoxic.sample(TOTAL_TOXIC))
    # prepared_65 = prepared_65.sample(frac=1).reset_index(drop=True)

    # random df
    random_df = train_sample.sample(n=TOTAL_SAMPLES, random_state=1008)

    # assert len(prepared_35) == len(prepared_50) == len(prepared_60) == len(prepared_65)

    print(f"Rebalanced dfs created...")

    return all_rebalanced + [random_df]




def clean_text(text, stop_ws=stops, stemmer=ps, str_output=True):

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



def get_metrics(output, should_print=True, detailed = False, label_col="y_test", score_col="predicted"):
    '''
    This function returns the model's metrics.

    '''
    metrics = {}
    targets = output[output[label_col] == 1]
    nontargets = output[output[label_col] == 0]

    dfs = [output, targets, nontargets]
    labels = ["Overall", "Target", "Non-Target"]

    for i in range(len(dfs)):

        df, label = dfs[i], labels[i]
        num_in_sample = df.shape[0]
        if label == "Non-Target":
            pos_label = 0
        else:
            pos_label = 1

        metrics[label] = {}

        accuracy = accuracy_score(df[label_col], df[score_col])
        metrics[label]['Accuracy'] = accuracy

        precision = precision_score(df[label_col], df[score_col], pos_label=pos_label)
        metrics[label]['Precision'] = precision

        recall = recall_score(df[label_col], df[score_col], pos_label=pos_label)
        metrics[label]['Recall'] = recall

        f1 = f1_score(df[label_col], df[score_col], pos_label=pos_label)
        metrics[label]['F1'] = f1

        if label == "Overall":
            roc_auc = round(roc_auc_score(df[label_col], df[score_col]), round_to)
            metrics[label]['ROC_AUC'] = roc_auc

        if should_print == True:
            print("Group Size: {}".format(num_in_sample))
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
            df, label = detail_dfs[i], detail_labels[i]
            num_in_sample = df.shape[0]
            metrics[label] = {}

            f1 = round(f1_score(df[label_col], df[score_col], pos_label=pos_label), round_to)
            metrics[label]['F1'] = f1

            if should_print == True:

                print("{} Samples: {}\nF1 Score: {}".format(label, num_in_sample, f1))

    return metrics


class TextData:
    def __init__(self, df, text_col='cleaned_no_stem'):
        # pull relevant data from df
        self.preprocessed_text = [word_list for word_list in df[text_col] ]

        # gather vocabulary corpus to store all words in training data
        # for i, comment in enumerate(self.preprocessed_text):
        #     if isinstance(comment, float):
        #         print("Comment num:",i, comment)
        self.vocab = Counter([word for comment in self.preprocessed_text
                              for word in comment]
                            ).most_common(VOCAB_SIZE-1)

        # word to index mapping
        self.word_to_idx = {k[0]: v+1 for v, k in
                            enumerate(self.vocab)}
        # all the unknown words will be mapped to index 0
        self.word_to_idx["UNK"] = 0
        self.idx_to_word = {v:k for k, v in self.word_to_idx.items()}
        self.label_to_idx = {TOXIC_LABEL: 1, NOT_TOXIC_LABEL: 0}
        self.idx_to_label = [NOT_TOXIC_LABEL, TOXIC_LABEL]
        self.vocab = set(self.word_to_idx.keys())



class LSTMModel(nn.Module):
    def __init__(self, X_data, y_data, test_X, test_y, hidden_dim, batch_size=1,
                 embed_dim=6, weight_decay=0, optimizer_fcn='Adam',
                 learning_rate=1e-3, num_layers=2, dropout=0.05, num_classes=2):
        super(LSTMModel, self).__init__()
        nn.Module.__init__(self)
        TextData.__init__(self, X_data)
        self.vocab_size = VOCAB_SIZE
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.output_dim = num_classes
        self.loss_fcn = nn.NLLLoss()
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.highest_f1 = -math.inf

        # Layer 1: Embedding Layer
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)

        # Layer 2: LSTM Layer
        self.lstm = nn.LSTM(input_size = self.embed_dim, hidden_size = self.hidden_dim,
                            num_layers = self.num_layers, dropout = self.dropout, batch_first=True)

        # Layer 3 (Output Layer): Linear
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

        # define optimizer
        if optimizer_fcn == 'Adam':
            self.optimizer = optim.Adam(params=self.parameters(),
                                                 weight_decay=self.weight_decay,
                                                 lr=self.learning_rate)
        elif optimizer_fcn == 'RMSprop':
            self.optimizer = optim.RMSprop(params=self.parameters(),
                                                 weight_decay=self.weight_decay,
                                                 lr=self.learning_rate)
        elif optimizer_fcn == 'SDG':
            self.optimizer = optim.SGD(params=self.parameters(),
                                                 weight_decay=self.weight_decay,
                                                 lr=self.learning_rate)

    def forward(self, input_seq):

        embed_out = self.embedding(input_seq)

        lstm_out, (hn, cn) = self.lstm(embed_out)

        out = F.log_softmax(self.linear(hn), dim=self.output_dim)

        return out


    def get_vectors(self, labels, text=None, text_col=None):
        X = []
        if text is None:
            text = self.preprocessed_text
        else:
            text = text[text_col]
        for comment in text:
            X.append(
                torch.tensor([self.word_to_idx.get(w, 0) for w in comment])
            )
        X_tensors = pad_sequence(X, batch_first=True)
        y_tensors = pd.get_dummies(labels).values
        y_tensors = torch.LongTensor(y_tensors)

        return X_tensors, y_tensors



    def classify(self, X_vec):
        '''
        This function classifies documents into their categories.
        docs are documents without labels.
        '''
        # pass forwrard w/o params --> get two items.
        # first value is representing positive value
        # second value is the negative value
        # bigger of the 2 should be the classification

        if USE_CUDA:
            X_vec = X_vec.cuda()

        argmaxes = []
        for vec in X_vec:
            results = self.forward(vec.unsqueeze(0))

            _, indices = torch.max(results.view(-1, results.size()[-1]), argmax=1)

        # indicies works cleanest --> for multiclass, look up in self.label_to_idx
            argmaxes.append(indices.item())

        return argmaxes



    def evaluate_classifier(self, validation_X, validation_y, text_col=None):
        '''
        This function evaluates the data with the current model.
        data contains both documents and labels.
        It calls classify() to make predictions,
        and compares with the correct labels to return
        the model accuracy on "data".
        '''
        X_vec, y_vec = self.get_vectors(validation_y, text=validation_X,
                                        text_col='cleaned_no_stem')
        if USE_CUDA:
            X_vec = X_vec.cuda()
            y_vec = y_vec.cuda()

        labels = [(single_tensor[1]).unsqueeze(0).item() for single_tensor in y_vec]

        classifications = self.classify(X_vec)

        accuracy = accuracy_score(labels, classifications)

        precision = precision_score(labels, classifications)

        recall = recall_score(labels, classifications)

        auc_roc = roc_auc_score(labels, classifications)

        f1 = f1_score(labels, classifications)

        results_df = validation_X.copy()
        results_df['predicted_score'] = classifications
        results_df['y_true'] = labels
        results_df['accuracy'] = results_df.predicted_score == results_df.y_true

        return f1, results_df


    def run_model(self, y_data, test_X, test_y, num_epochs, loss_record, text_col=None, savestate=False):
        results = []
        X_vec, y_vec = self.get_vectors(y_data, text=None)

        if USE_CUDA:
            X_vec = X_vec.cuda()
            y_vec = y_vec.cuda()


        last_mean_error = -math.inf
        num_training_samples = X_vec.size()[0]

        print(f"Epochs: {num_epochs}; Train Size: {num_training_samples}; Test Size: {test_X.shape[0]})")

        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch}...")
            for i in range(num_training_samples):
                # Zero out gradient, else they will accumulate between epochs
                self.optimizer.zero_grad()

                y_pred = self.forward(X_vec[i].unsqueeze(0))

                loss = self.loss_fcn(input=y_pred.view(-1, y_pred.size()[-1]),
                                     target=(y_vec[i][1]).unsqueeze(0))

                # Backward pass
                loss.backward()

                # Update parameters
                self.optimizer.step()

            thresholds: {
                5000: 1000,
                1000: 100,
                100: 10,
                20: 5,
                10: 3
            }

            print_at = 2
            if num_epochs in  thresholds.keys():
                print_at = thresholds[num_epochs]
            # print results for every few epochs
            if epoch % print_at == 0:
                print(f"Epoch {epoch}, Negative Log Linear Loss: {loss.item()}")
                loss_record[epoch] = loss.item()

                with torch.no_grad():
                    if np.mean(np.abs(loss.item())) < last_mean_error:
                        print(f"Delta after {epoch} iterations: {np.mean(np.abs(loss.item()))}")
                        last_mean_error = np.mean(np.abs(loss.item()))
                    else:
                        if last_mean_error > -math.inf:
                            print(f"Break: {np.mean(np.abs(loss.item()))} > {last_mean_error}")
                            break

            print()
            print("Starting Evaluation")

            model_f1, results_df = self.evaluate_classifier(test_X, test_y, text_col=text_col)
            print(f"Epoch {epoch} F1 Score: {model_f1}")
            # pickle results df after every epoch
            results_df.to_pickle(savestate + f'_epoch{epoch}.pkl')

            best_model_state_dict = None

            if model_f1 > self.highest_f1:
                self.highest_f1 = model_f1
                self.best_epoch = epoch
                best_model_state_dict = self.state_dict()

        print(f"Highest F1 Score: {self.highest_f1}, Epoch: {self.best_epoch}")

        if savestate:
            savepath = savestate + f'_epoch{highest_epoch}.pt'
            torch.save(best_model_state_dict, savepath)

        return results_df, best_model_state_dict



def main(filepath="jigsaw_toxic/"):

    # df = load_label(filepath)
    #
    # df['cleaned_no_stem'] = df["comment_text"].apply(clean_text,args=(stops,None,False),)
    #
    # rebalance_dict = {0: 35, 1: 50, 2: 60, 3: 65, 4: .75, 5: 'random'}
    #
    # data_proportions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
    #
    # test_ratio = 0.2
    #
    # for p, proportion in enumerate(data_proportions):
    #
    #     train_sample, val_set, test_set = get_samples(df, proportion=proportion, train_test_ratio=(1-test_ratio))
    #
    #     prepared_35, prepared_50, prepared_60, prepared_65, prepared_75, random_df = rebalance_data(train_sample)
    #
        # for i, p_df in enumerate([prepared_35, prepared_50, prepared_60, prepared_65, prepared_75, random_df]):
        #     model_name= f'{int(data_proportions[p]*100)}pct_model_{rebalance_dict[i]}toxic'
        #     val_set.to_pickle("jigsaw_toxic/train_test/" + model_name + "_val.pkl")
    #         test_set.to_pickle("jigsaw_toxic/train_test/" + model_name + "_test.pkl")
    #         p_df.to_pickle("jigsaw_toxic/train_test/" + model_name + "_train.pkl")

    filelist = []
    for file in os.listdir(filepath + "train_test/"):
        if file.endswith(".pkl"):
            if "_test" not in file:
                filelist.append(file)

    filelist.sort()

    train_list, val_list = [], []
    for x in filelist:
        (train_list if "_train" in x else val_list).append(x)

    for p_df, val_set in zip(train_list, val_list)
        model_name = os.path.splitext(p_df)[0].replace("_train", "")
        p_df = pd.read_pickle(filepath + p_df)
        val_set = pd.read_pickle(filepath + val_set)

        print(f"{model_name}:")
        X_train = p_df.drop('label', axis=1)
        y_train = p_df['label']
        test_sample = val_set.sample( n=math.ceil(len(X_train)*test_ratio), random_state=1008 )
        # test_sample = val_set.sample(frac=test_ratio, replace=True)
        X_test = test_sample.drop('label', axis=1)
        y_test = test_sample['label']

        lstm_model = LSTMModel(X_train, y_train,
                                  X_test, y_test, hidden_dim=50,
                                  num_layers=1, embed_dim=50, batch_size=1,
                                  dropout=0, num_classes=2)
        if USE_CUDA:
            lstm_model = lstm_model.cuda()

        lstm_model.train()

        NUM_EPOCHS = 6
        hist_lstm = np.zeros(NUM_EPOCHS)

        _, model_state_dict = lstm_model.run_model(
            y_train, X_test, y_test, NUM_EPOCHS, hist_lstm, text_col='cleaned_no_stem',
            savestate=model_name)

        print(model_state_dict)
        # results dfs pickled after every epoch, can scp .pkl files and check metrics locally
        # metric_dict = get_metrics(results, detailed=True,
        #             label_col="y_true", score_col="predicted_score")
        #
        # metric_dict.to_pickle(model_name+"_metric_dict.pkl")

def reload_model(state_file, args, kwargs):
    lstm_instance = LSTMModel(*args, **kwargs)
    lstm_instance.load_state_dict(torch.load(state_file))
    lstm_instance.eval()

    return lstm_instance


if __name__ == '__main__':
    class Args():
        pass

    a = Args()
    parser = argparse.ArgumentParser(description="Collect arguments for running model")

    parser.add_argument('--infile', help="Zipped file from which to read from.")

    try:
        args = parser.parse_args(namespace=a)
    except argparse.ArgumentError or argparse.ArgumentTypeError as exc:
        sys.exit("exec_lstm error: Please review arguments passed: {}".format(
           args, exc.message))
    except Exception as e:
        sys.exit("exec_lstm error: Please review arguments passed: {}".format(e))

    # try:
    if a.infile:
        filename = a.infile
        main(filename)
    # except Exception as e:
    #     # check for any exceptions not covered above
    #     sys.exit("exec_lstm error: An unexpected error occurred when processing "
    #         "your request: {}".format(e))