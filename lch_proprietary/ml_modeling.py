import pandas as pd
import numpy as np
import itertools
from itertools import chain
import sklearn
from sklearn import preprocessing, svm, metrics, tree, decomposition
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, f1_score, roc_auc_score
import magiclooping as mp



def split_data(df, outcome_var, geo_columns, test_size, seed = None):
    '''
    Separate data frame into training and test subsets based on specified size
    for model training and evaluation.

    Inputs:
        df: pandas dataframe
        outcome_var: (string) variable model will predict
        geo_columns:  (list of strings) list of column names corresponding to
            columns with numeric geographical information (ex: zipcodes)
        test_size: (float) proportion of data to hold back from training for
            testing

    Output: testing and training data sets for predictors and outcome variable
    '''
    # remove outcome variable and highly correlated variables
    all_drops = [outcome_var] + geo_columns
    X = df.drop(all_drops, axis=1)
    # isolate outcome variable in separate data frame
    Y = df[outcome_var]

    return train_test_split(X, Y, test_size = test_size, random_state = seed)


def temporal_train_test_split(df, outcome_var, exclude = [], keep_cols = False):
    if not keep_cols:
        skips = [outcome_var] + exclude
        Xs = df.drop(skips, axis = 1)
    else:
        Xs = df[keep_cols]

    Ys = df[outcome_var]

    return Xs, Ys

def get_classifier_params(grid_size):
    '''
    Returns initialized classifiers and paramters to loop over for three sizes of grids:
    Thanks to the Data Science for Social Good team for the recommendations!

    Mini - to test a small number of classifiers across a few a few different 
        parameter mixes
    Test - to test just one variation of each classifier
    Small - to test key classifiers across a few different parameter mixes
    Large - for robust modeling with a large number of key classifiers
    '''

    if grid_size == 'mini':
        clfs = {
            'LogisticRegression': LogisticRegression(penalty='l1', C=1e5),
            'DecisionTree': DecisionTreeClassifier(random_state=1008),
            'KNN': KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
            'SGD': SGDClassifier(loss="hinge", penalty="l2")
        }

        params_dict = {
        
            'LogisticRegression': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
            'DecisionTree': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
            "RandomForest":{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1], 'random_state':[1008]},
            'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
            "SGD": { 'loss': ['perceptron'], 'penalty': ['l1', 'l2']}

        }

    else:

        clfs = {
        'DecisionTree': DecisionTreeClassifier(random_state=1008),
        'LogisticRegression': LogisticRegression(penalty='l1', C=1e5),
        'Bagging': BaggingClassifier(base_estimator=LogisticRegression(
            penalty='l1', C=1e5)),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=1008),
        'AdaBoost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
            algorithm="SAMME", n_estimators=200),
        'GradientBoosting': GradientBoostingClassifier(learning_rate=0.05,
            subsample=0.5, max_depth=6, n_estimators=10),
        'NaiveBayes': GaussianNB(),
        "RandomForest": RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'KNN': KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
        }

        if grid_size == 'test':

            params_dict = {
                "DecisionTree": {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10], 'random_state':[1008]},
                "LogisticRegression": { 'penalty': ['l1'], 'C': [0.01]},
                "SGD": { 'loss': ['perceptron'], 'penalty': ['l2']},
                "ExtraTrees": { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10], 'random_state':[1008]},
                "SVM" :{'C' :[0.01],'kernel':['linear']},
                "AdaBoost": { 'algorithm': ['SAMME'], 'n_estimators': [1]},
                "GradientBoosting": {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
                "NaiveBayes": {},
                "RandomForest":{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10], 'random_state':[1008]},
                "KNN":{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
               }
    
        if grid_size == 'small':

            params_dict =  { 
                "DecisionTree": {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10], 'random_state':[1008]},
                "LogisticRegression": { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
                "Bagging": {}
                "SGD": { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
                "ExtraTrees": { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
                "SVM":{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
                "AdaBoost": { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
                "GradientBoosting": {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
                "NaiveBayes": {},
                "RandomForest":{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1], 'random_state':[1008]},
                "KNN":{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
               }
        elif grid_size == 'medium':

            params_dict = {
                "DecisionTree": {'criterion': ['gini', 'entropy'],
                    'max_depth': [1,5,10,20,50,100],
                    'max_features': [None, 'sqrt','log2'],
                    'min_samples_split': [2,5,10], 'random_state':[1008]},
                "LogisticRegression": { 'penalty': ['l1'], 'C': [0.01]},
                "Bagging": {},
                "SVM": {'C' :[0.01],'kernel':['linear']},
                "AdaBoost": { 'algorithm': ['SAMME'], 'n_estimators': [1]},
                'GradientBoosting': {'n_estimators': [1],
                    'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
                'NaiveBayes' : {},
                "RandomForest": {'n_estimators': [100, 10000],
                    'max_depth': [5,50], 'max_features': ['sqrt','log2'],
                    'min_samples_split': [2,10], 'n_jobs':[-1],
                    'random_state':[1008]},
                "KNN": {'n_neighbors': [5],'weights': ['uniform'],
                    'algorithm': ['auto']}
        }

        elif grid_size == 'large':

            params_dict = { 
                "DecisionTree": {'criterion': ['gini', 'entropy'],
                    'max_depth': [1,5,10,20,50,100],
                    'max_features': [None, 'sqrt','log2'],
                    'min_samples_split': [2,5,10], 'random_state':[1008]},
                "LogisticRegression": { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10] },
                "Bagging": {},
                "SGD": {'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
                "ExtraTrees": {'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
                "SVM": {'C':[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
                "AdaBoost": { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
                "GradientBoosting": {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
                "NaiveBayes": {},
                "RandomForest": {'n_estimators': [1,10,100,1000,10000],
                    'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],
                    'min_samples_split': [2,5,10], 'n_jobs': [-1],'random_state':[1008]},
                "KNN": {'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
            }
    
    return clfs, params_dict

def develop_args(name, params_dict):
        # create dictionaries for each possible tuning option specified
        # in param_dict

    options = params_dict[name]
    tuners = list(options.keys())
    list_params = list(itertools.product(*options.values()))

    all_model_params = []

    for params in list_params:
        kwargs_dict = dict(zip(tuners, params))
        all_model_params.append(kwargs_dict)

    return all_model_params



def clf_loop(X_train, y_train, X_test, y_test, set_num, grid_size='mini',
    ks=[100, 5, 10, 20], plot=False, select_clfs=None):
    '''
    Attribution: Adapted from Rayid Ghani's magicloop and simpleloop examples
    https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    result_cols = ['set_num', 'model_type','clf', 'parameters',
                    'baseline_precision','baseline_recall','auc-roc']

    # define columns for metrics at each threshold specified in function call
    result_cols += list(chain.from_iterable(('p_at_{}'.format(threshold),
        'r_at_{}'.format(threshold),
        'f1_at_{}'.format(threshold)) for threshold in ks))

    # define dataframe to write results to
    results_df =  pd.DataFrame(columns=result_cols)

    clfs, params_dict = get_classifier_params(grid_size=grid_size)

    if select_clfs:
        clfs = {clf: clfs.get(clf, None) for clf in select_clfs}
        params_dict = {clf: params_dict.get(clf, None) for clf in select_clfs}


    for name, clf in clfs.items():
        print("Creating classifier: {}".format(name))

        if clf is None:
            continue

        # create all possible models using tuners in dictionaries created above
        all_model_params = develop_args(name, params_dict)

        for args in all_model_params:
            try:
                clf.set_params(**args)
                y_scores = clf.fit(X_train,
                    y_train.values.ravel()).predict_proba(X_test)[:,1]

                y_scores_sorted, y_true_sorted = joint_sort_descending(
                    np.array(y_scores), np.array(y_test))

                # print("Evaluating {} models".format(name))
                precision_100, recall_100, _ = scores_at_k(y_true_sorted, y_scores_sorted, 100.0)

                results_list = [set_num, name, clf, args, precision_100 recall_100,
                    roc_auc_score(y_test, y_scores)]

                for threshold in ks:
                    precision, recall, f1 = scores_at_k(y_true_sorted,
                        y_scores_sorted, threshold)
                    results_list += [precision, recall, f1]

                results_df.loc[len(results_df)] = results_list

                if plot:
                    plot_precision_recall_n(y_test, y_scores, clf)

            except Exception as e:
                print(f"Error {e} on model {name} with parameters {args}")
                print()
                continue

    return results_df



def temporal_train_test_split(df, outcome_var, exclude = [], subset_cols = False):
    if not subset_cols:
        skips = [outcome_var] + exclude
        Xs = df.drop(skips, axis = 1)
    else:
        Xs = df[subset_cols]

    Ys = df[outcome_var]

    return Xs, Ys


def run_models(train_test_tuples, outcome_var, clfs, ks = [5, 10, 20]):
    all_results = []
    for i, (train, test) in enumerate(train_test_tuples):
        print("set", i)

        # x_train, y_train = temporal_train_test_split(train, outcome_var)
        # x_test, y_test = temporal_train_test_split(test, outcome_var)
        # results = cf_loop(x_train, y_train, x_test, y_test,
        #                      ks = ks,
        #                      set_num = i, params_dict = None,
        #                      which_clfs = clfs)
        # all_results.append(results)

    return pd.concat(all_results, ignore_index = True)


def construct_best(metrics_df):
    identifiers = ['clf', 'parameters', 'model_type', 'set_num']
    metric_cols = set(metrics_df.columns) - set(identifiers)

    best_df = pd.DataFrame(columns = ['metric','baseline_p', 'max_value',
                                     'model_num', 'model_type', 'clf',
                                     'test_set'])

    for col in metric_cols:
        best = metrics_df[col].max()
        idx = metrics_df[col].idxmax()
        row = [col, metrics_df.loc[idx, 'baseline_precision'],
               best, idx, metrics_df.loc[idx, 'model_type'],
               metrics_df.loc[idx, 'clf'], metrics_df.loc[idx, 'set_num']]

        best_df.loc[len(best_df)] = row

    best_df.set_index('metric', inplace = True)
    return best_df



def loop_dt(param_dict, X_train, X_test,
                y_train, y_test):
    '''
    Loop over series of possible parameters for decision tree classifier to
    train and test models, storing accuracy scores in a data frame

    Inputs:
        param_dict: (dictionary) possible decision tree parameters
        X_train: data set of predictor variables for training
        X_test: data set of predictor variables for testing
        y_train: outcome variable for training
        y_test: outcome variable for testing

    Outputs:
        accuracy_df: (data frame) model parameters and accuracy scores for
            each iteration of the model

    Attribution: adapted combinations of parameters from Moinuddin Quadri's
    suggestion for looping: https://stackoverflow.com/questions/42627795/i-want-to-loop-through-all-possible-combinations-of-values-of-a-dictionary
    and method for faster population of a data frame row-by-row from ShikharDua:
    https://stackoverflow.com/questions/10715965/add-one-row-in-a-pandas-dataframe
    '''
    rows_list = []
    for clf_type, classifier in classifier_type.items():

        for params in list(itertools.product(*param_dict.values())):
            classifier(params)
            dec_tree.fit(X_train, y_train)


    rows_list = []
    for params in list(itertools.product(*param_dict.values())):
        dec_tree = DecisionTreeClassifier(criterion = params[0],
                                          max_depth = params[1],
                                          max_features = params[2],
                                          min_samples_split = params[3])
        dec_tree.fit(X_train, y_train)

        train_predictions = dec_tree.predict(x_train)
        test_predictions = dec_tree.predict(X_test)

        # evaluate accuracy
        train_acc = accuracy(train_predictions, y_train)
        test_acc = accuracy(test_predictions, y_test)

        acc_dict = {}
        (acc_dict['criterion'], acc_dict['max_depth'], acc_dict['max_features'],
        acc_dict['min_samples_split']) = params
        acc_dict['train_acc'] = train_acc
        acc_dict['test_acc'] = test_acc

        rows_list.append(acc_dict)

    accuracy_df = pd.DataFrame(rows_list)

    return accuracy_df


def create_best_tree(accuracy_df, X_train, y_train):
    '''
    Create decision tree based on highest accuracy score in model testing, to
    view feature importance of each fitted feature

    Inputs:
        accuracy_df: (data frame) model parameters and accuracy scores for
            each iteration of the model
        X_train: data set of predictor variables for training
        y_train: outcome variable for training

    Outputs:
        best_tree: (classifier object) decision tree made with parameters used
            for highest-ranked model in terms of accuracy score during
            parameters loop
    '''
    accuracy_ranked = accuracy_df.sort_values('test_acc', ascending = False)
    dec_tree = DecisionTreeClassifier(
    criterion = accuracy_ranked.loc[accuracy_ranked.iloc[0].name, 'criterion'],
    max_depth = accuracy_ranked.loc[accuracy_ranked.iloc[0].name, 'max_depth'],
    max_features = accuracy_ranked.loc[accuracy_ranked.iloc[0].name,
                                        'max_features'],
    min_samples_split = accuracy_ranked.loc[accuracy_ranked.iloc[0].name,
                                            'min_samples_split'])

    dec_tree.fit(X_train, y_train)

    return dec_tree



def feature_importance_ranking(best_tree, X_train):
    '''
    View feature importance of each fitted feature

    Inputs:
        best_tree: (classifier object) decision tree made with parameters used
            for highest-ranked model in terms of accuracy score during
            parameters loop

    Outputs:
        features_df: (data frame) table of feature importance for each
        predictor variable
    '''
    features_df = pd.DataFrame(best_tree.feature_importances_,
                                X_train.columns).rename(
                                columns = {0: 'feature_importance'},
                                inplace = True)
    features_df.sort_values(by = 'feature_importance', ascending = False)
    return features_df


def visualize_best_tree(best_tree, X_train):
    '''
    Visualize decision tree object with GraphWiz
    '''
    viz = sklearn.tree.export_graphviz(best_tree,
                    feature_names = X_train.columns,
                    class_names=['Financially Stable', 'Financial Distress'],
                    rounded=False, filled=True)

    with open("tree.dot") as f:
        dot_graph = f.read()
        graph = graphviz.Source(dot_graph)

    return graph


def scores_at_k(y_true, y_scores, k):
    '''
    Calculate precision, recall, and f1 score at a given threshold
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    precision = precision_score(y_true_sorted, preds_at_k)
    recall = recall_score(y_true_sorted, preds_at_k)
    f1 = f1_score(y_true_sorted, preds_at_k)
    return precision, recall, f1



def generate_binary_at_k(y_scores_sorted, k):
    '''
    Attribution: Adapted from Rayid Ghani magicloops 
    https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    cutoff_index = int(len(y_scores_sorted) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores_sorted))]
    return predictions_binary


def joint_sort_descending(l1, l2):
    '''
    Attribution: Adapted from Rayid Ghani magicloops
    https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]


def plot_precision_recall_n(testing_outcome, test_pred, model_name):
    '''
    Attribution: Adapted from Rayid Ghani magicloops 
    https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    from sklearn.metrics import precision_recall_curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(testing_outcome, test_pred)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(test_pred)
    for value in pr_thresholds:
        num_above_thresh = len(test_pred[test_pred>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('Percentage of Population')
    ax1.set_ylabel('Precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('Recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()
