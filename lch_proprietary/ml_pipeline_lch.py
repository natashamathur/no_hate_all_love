import re
import logging
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, normalize

logging.getLogger(__name__)


def retrieve_data(filename, headers = False, set_ind = None):
    '''
    Read in data from CSV to a pandas dataframe

    Inputs:
        filename: (string) filename of CSV
        headers: (boolean) whether or not CSV includes headers
        ind: (integer) CSV column number of values to be used as indices in 
            data frame

    Output: pandas data frame
    '''
    if headers and isinstance(set_ind, int):
        data_df = pd.read_csv(filename, header = 0, index_col = set_ind)
    elif headers and not set_ind:
        data_df = pd.read_csv(filename, header = 0)
    else:
        data_df = pd.read_csv(filename)
    return data_df



def print_null_freq(df, blanks_only = False):
    '''
    For all columns in a given dataframe, calculate and print number of null and non-null values

    Attribution: Adapted from https://github.com/yhat/DataGotham2013/blob/master/analysis/main.py
    '''
    df_lng = pd.melt(df)
    null_variables = df_lng.value.isnull()
    all_rows = pd.crosstab(df_lng.variable, null_variables)
        
    if blanks_only:
        try:
            return all_rows[all_rows[True] > 0]
        except:
            return False
    else: 
        return all_rows


def still_blank(train_test_tuples):
    '''
    Check for remaining null values after dummy variable creation is complete.
    '''
    to_impute = []
    for train, test in train_test_tuples:
        with_blanks = print_null_freq(train, blanks_only = True)
        print(with_blanks)
        print()
        to_impute.append(list(with_blanks.index))
    return to_impute



def create_col_ref(df):
    '''
    Develop quick check of column position via dictionary
    '''
    col_list = df.columns
    col_dict = {}
    for list_position, col_name in enumerate(col_list):
        col_dict[col_name] = list_position
    return col_dict



def abs_diff(col, factor, col_median, MAD):
    '''
    Calculate modified z-score of value in pandas data frame column, using 
    sys.float_info.min to avoid dividing by zero

    Inputs:
        col: column name in pandas data frame
        factor: factor for calculating modified z-score (0.6745)
        col_median: median value of pandas data frame column
        MAD: mean absolute difference calculated from pandas dataframe column
    
    Output: (float) absolute difference between column value and column meaan 
        absolute difference

    Attribution: workaround for MAD = 0 adapted from https://stats.stackexchange.com/questions/339932/iglewicz-and-hoaglin-outlier-test-with-modified-z-scores-what-should-i-do-if-t
    '''
    if MAD == 0:
        MAD = 2.2250738585072014e-308 
    return (x - y)/ MAD



def outliers_modified_z_score(df, col):
    '''
    Identify outliers (values falling outside 3.5 times modified z-score of 
    median) in a column of a given data frame

    Output: (pandas series) outlier values in designated column

    Attribution: Modified z-score method for identifying outliers adapted from 
    http://colingorrie.github.io/outlier-detection.html
    '''
    threshold = 3.5
    zscore_factor = 0.6745
    col_median = df[col].astype(float).median()
    median_absolute_deviation = abs(df[col] - col_median).mean()
    
    modified_zscore = df[col].apply(lambda x: abs_diff(x, zscore_factor, 
                                    col_median, median_absolute_deviation))
    return modified_zscore[modified_zscore > threshold]



def convert_dates(date_series):
    '''
    Faster approach to datetime parsing for large datasets leveraging repated dates.

    Attribution: https://github.com/sanand0/benchmarks/commit/0baf65b290b10016e6c5118f6c4055b0c45be2b0
    '''
    dates = {date:pd.to_datetime(date) for date in date_series.unique()}
    return date_series.map(dates)


# def make_boolean(df, cols, value_1s, ints = True):
#     if ints:
#         true_val = 1
#         neg_val = 0
#     else:
#         true_val = True
#         neg_val = False
        
#     for col in cols:
#         df.loc[df[col] != value_1s, col] = neg_val
#         df.loc[df[col] == value_1s, col] = true_val


def make_boolean(df, cols, true_val, ints = True):
    for col in cols:
        df.loc[:, col] =  df[col] == true_val

        if ints:
            df.loc[:, col] = df[col].astype('int')




def likely_outliers(df, top=True, pct_change=True):
    '''
    View top or bottom 10% of values (or percent change between percentiles) 
    in each column of a given data frame

    Inputs: 
        df: pandas dataframe
        top: (boolean) indicator of whether to return top or bottom values
        pct_change: whether to return aboslute max/ min values or percent change

    Output: (dataframe) 'pct_change' is True - percent changes between 
        values at each 100th of a percentile for top or bottom values 
        in given dataframe column OR

        (dataframe) 'pct_change' is False - values at each 100th of a 
        percentile for top or bottom values dataframe column 
    '''
    if top:
        if pct_change:
            return df.quantile(q=np.arange(0.9, 1.001, 0.001)).pct_change()
        else:
            return df.quantile(q=np.arange(0.99, 1.001, 0.001))

    else: 
        if pct_change:
            return df.quantile(q=np.arange(0.0, 0.011, 0.001)).pct_change()
        else:
            return df.quantile(q=np.arange(0.0, 0.011, 0.001))



# def view_likely_outliers(df, max = True):
#     '''
#     View percent change between percentiles in top or bottom 10% of values in  
#     each column of a given data frame 

#     Inputs: 
#         df: pandas dataframe
#         max: (boolean) indicator of whether to return to or bottom values

#     Output: (dataframe) percent changes between values at each 100th of a 
#         percentile for top or bottom values in given dataframe column
#     '''
#     if max:
#         return df.quantile(q=np.arange(0.9, 1.001, 0.001)).pct_change()
#     else: 
#         return df.quantile(q=np.arange(0.0, 0.011, 0.001)).pct_change()



def remove_over_under_threshold(df, col, min_val = False, max_val = False, lower = None, upper = False):
    '''
    Remove values over given percentile or value in a column of a given data 
    frame
    '''
    if max_val:
        df.loc[df[col] > max_val, col] = None
    if min_val:
        df.loc[df[col] < min_val, col] = None
    if upper:
        maxes = likely_outliers(df, top=True, pct_change=False)
        # maxes = view_max_mins(df, top = True)
        df.loc[df[col] > maxes.loc[upper, col], col] = None
    if lower:
        # mins = view_max_mins(df, top = False)
        mins = likely_outliers(df, top=False, pct_change=False)
        df.loc[df[col] < mins.loc[lower, col], col] = None
    

def remove_dramatic_outliers(df, col, threshold, top = True):
    '''
    Remove values over certain level of percent change in a column of a given 
    data frame
    '''
    if top:
        # maxes = view_max_mins(df, top = True)
        # likely_outliers_upper = view_likely_outliers(df, top = True)
        maxes = likely_outliers(df, top=True, pct_change=False)
        uppoer_outliers = likely_outliers(df, top=True, pct_change=True)

        outlier_values = list(maxes.loc[
            uppoer_outliers[ uppoer_outliers[col] > threshold ][col].index, col])
    else: 
        # mins = view_max_mins(df, top = False)
        # lower_outliers = view_likely_outliers(df, top = False)
        mins = likely_outliers(df, top=False, pct_change=False)
        lower_outliers = likely_outliers(df, top=False, pct_change=True)

        outlier_values = list(mins.loc[
            lower_outliers[ lower_outliers[col] > threshold ][col].index, col])
    
    df = df[~df[col].isin(outlier_values)]



def basic_fill_vals(df, col_name, test_df = None, method = None, replace_with = None):
    '''
    For columns with more easily predicatable null values, fill with mean, median, or zero

    Inputs:
        df: pandas data frame
        col_name: (string) column of interest
        method: (string) desired method for filling null values in data frame. 
            Inputs can be "zeros", "median", or "mean"
    '''
    if method == "zeros":
        df[col_name].fillna(0, inplace = True)
    elif method == "replace":
        replacement_val = replace_with
        df[col_name].fillna(replacement_val, inplace = True)
    elif method == "median":
        replacement_val = df[col_name].median()
        df[col_name].fillna(replacement_val, inplace = True)
    elif method == "mean":
        replacement_val = df[col_name].mean()
        df[col_name].fillna(replacement_val, inplace = True)

    # if imputing train-test set, fill test data frame with same values
    if test_df is not None:
        test_df[col_name].fillna(replacement_val, inplace = True)



def check_col_types(df):
    return pd.DataFrame(df.dtypes, df.columns).rename({0: 'data_type'}, axis = 1)



def is_category(col_name, keywords=None, geos_indicator=True):
    '''
    Utility function to determine whether a given column name includes key words or
    phrases indicating it is categorical.

    Inputs:
        col_name: (string) name of a column
        keywords: (list) strings indicating a column is categorical or geographical 
        geos_indicator: (boolean) whether or not to include geographical words or phrases
            in column name search
    '''
    search_for = ["_bin","_was_null", "_id"]

    if keywords:
        search_for += keywords

    if geos_indicator:
        search_for += ["city", "state", "county", "country", "zip", "zipcode", "latitude", "longitude"]

    search_for = "|".join(search_for)

    return re.search(search_for, col_name)


def summarize_df(df):
    type_dict = defaultdict(list)
    geos = ["city", "state", "county", "country", "zip", "zipcode", "latitude", "longitude"]
    geos = "|".join(geos)
    summary = pd.DataFrame(columns = ["col_name", "num_values", "num_nulls", "unique_values",  "data_type", "col_type", "most_common", "prevalence"])
    
    for col in df.columns:
        num_values = df[col].value_counts().sum()
        uniques = len(df[col].unique())
        nulls = df[col].isnull().sum()
        most_common = list(df[col].mode())[0]
        mode_count = (df[col].value_counts().max() / num_values) * 100
        dtype = df[col].dtype


        if re.search(geos, col):
            col_type = "geo"
            type_dict["geo"].append(col)
        elif re.search("id|_id|ID", col):
            col_type = "ID"
            type_dict["ID"].append(col)
        elif df[col].dtype.str[1] == 'M':
            col_type = "datetime"
            type_dict["datetime"].append(col)
        elif uniques == 1 or uniques == 2:
            col_type = "binary"
            type_dict["binary"].append(col)
        elif df[col].dtype.kind in 'uifc':
            col_type = "numeric"
            type_dict["numeric"].append(col)
        elif uniques <= 6:
            col_type = "multi"
            type_dict["multi"].append(col)
        elif uniques > 6:
            col_type = "tops"
            type_dict["tops"].append(col)

        summary.loc[col] = [col, num_values, nulls, uniques, dtype, col_type, most_common, mode_count]
    
    summary.set_index("col_name", inplace = True)
    return summary, type_dict


def recateogrize_col(col, new_category, col_dict):
    for category, cols_list in col_dict.items():
        if col in cols_list:
            col_dict[category] = [column for column in cols_list if column != col]
    col_dict[new_category].append(col)
    return col_dict


def replace_dummies(df, cols_to_dummy):
    return pd.get_dummies(df, columns = cols_to_dummy , dummy_na=True)



def isolate_categoricals(df, categoricals_fcn=is_category, ret_categoricals = False, keywords = None, geos_indicator=True):
    '''
    Retrieve list of cateogrical or non-categorical columns from a given dataframe

    Inputs:
        df: pandas dataframe
        categoricals_fcn: (function) Function to parse column name and return boolean
            indicating whether or not column is categorical
        ret_categoricals: (boolean) True when output should be list of  
            categorical colmn names, False when output should be list of 
            non-categorical column names
        keywords: (list) strings indicating a column is categorical or geographical 
        geos_indicator: (boolean) whether or not to include geographical words or phrases
            in column name search

    Outputs: list of column names from data frame
    '''
    categorical = [col for col in df.columns if categoricals_fcn(col, keywords=keywords, geos_indicator=geos_indicator)]
    # non_categorical = [col for col in df.columns if not categoricals_fcn(col, flag = keyword, geos = geos_indicator)]
    non_categorical = [col for col in df.columns if col not in categorical]
    
    if ret_categoricals:
        return categorical
    else:
        return non_categorical



def change_col_name(df, current_name, new_name):
    '''
    Change name of a single column in a given data frame
    '''
    # df.columns = [new_name if col == current_name else col for col in df.columns]
    df.rename(columns={current_name:new_name}, inplace=True)



def drop_unwanted(df, drop_list):
    df.drop(drop_list, axis = 1, inplace = True)



# def time_series_split(df, date_col, train_size, test_size, increment = 'month', specify_start = None):
#     '''
#     Pass start date as 'YYYY-MM-DD'
#     Increment options: ['day', 'month', 'year']
#     Train/ test size should be integers corresponding to the increment
#     '''
#     if specify_start:
#         min_date = datetime.strptime(specify_start, '%Y-%m-%d')
#     else:
#         min_date = df[date_col].min()

#         if min_date.day > 25:
#             min_date += datetime.timedelta(days = 7)
#             min_date = min_date.replace(day=1, hour=0, minute=0, second=0)

#         else:
#             min_date = min_date.replace(day=1, hour=0, minute=0, second=0)

    
#     if increment == 'month':
#         train_max = min_date + relativedelta(months = train_size) - timedelta(days = 1)
#         test_min = train_max + timedelta(days = 1)
#         test_max = min(test_min + relativedelta(months = test_size), df[date_col].max())

#     if increment == 'day':
#         train_max = min_date + relativedelta(days = train_size)
#         test_min = train_max + timedelta(days = 1)
#         test_max = min((test_min + relativedelta(days = test_size)), df[date_col].max())
    
#     if increment == 'year':
#         train_max = timedelta(months = train_size) - timedelta(days = 1)
#         test_min = train_max + relativedelta(years = train_size)
#         test_max = min(test_min + relativedelta(years = test_size), df[date_col].max())

#     new_df = df[df.columns]
#     train_df = new_df[(new_df[date_col] >= min_date) & (new_df[date_col] <= train_max)]
#     test_df = new_df[(new_df[date_col] >= test_min) & (new_df[date_col] <= test_max)]
    
#     date_refs = (increment, min_date, train_size, test_min, test_size)

#     return train_df, test_df, date_refs



def time_series_split(df, date_col, train_size, test_size, increment = 'months', specify_start = None):
    '''
    Pass start date ('specify_start') as 'YYYY-MM-DD'
    Increment options: ['days', 'weeks', months', 'years']
    Train/ test size should be integers corresponding to increment
    '''
    if specify_start:
        min_date = datetime.strptime(specify_start, '%Y-%m-%d')
    else:
        min_date = df[date_col].min()

    # start from end of test sent and build from there
    test_max = df[date_col].max()

    # need to have at least test_size amount of months in test_set
    # test_min = test_max - relativedelta(months = test_size)
    test_min = test_max - relativedelta(**{increment: test_size})
    
    # if df max date minus test_size is outside the dataset (or specified 
    # subset), raise error
    if test_min <= min_date + timedelta(days = 1):
        train_interval = relativedelta(train_max, train_min)
        raise ValueError(f"Sspecified test_size '{test_size}' results in test set start date before minimum date '{min_date.strftime('%Y-%m-%d')}.' Please update parameters.")
    
    else:
        # start training set the day before test_min date
        train_max = test_min - timedelta(days = 1)

        # train_min = max(train_max - relativedelta(months = train_size), min_date)
        train_min = train_max - relativedelta(**{increment: train_size})

        if train_min < min_date:
            dataset_interval = relativedelta(test_max, train_min)
            interval_string = f"{dataset_interval.years} year(s), {dataset_interval.months} month(s), {dataset_interval.days} day(s)"

            abs_min = df[date_col].min()
            logging.warning(f"Sspecified train_size '{train_size}' {increment} and test_size '{test_size}' {increment} exceed specified dataset interval '{interval_string}' beginning at minimum date '{min_date.strftime('%Y-%m-%d')}.' Creating test set with start date '{abs_min.strftime('%Y-%m-%d')}.'")
            train_min = abs_min

    train_df = df.loc[ (df[date_col] >= train_min) & (df[date_col] <= train_max), df.columns]
    test_df = df.loc[ (df[date_col] >= test_min) & (df[date_col] <= test_max), df.columns]
    
    date_refs = (increment, train_min, train_max, test_min, test_max)

    return train_df, test_df, date_refs



def create_expanding_splits(df, total_periods, date_col, train_period_size, 
                            test_period_size, increment = 'months', 
                            specify_start = None):
    '''
    Pass start date ('specify_start') as 'YYYY-MM-DD'
    Increment options: ['days', 'weeks', months', 'years']
    Train/ test size should be integers corresponding to increment
    '''
    if specify_start:
        min_date = datetime.strptime(specify_start, '%Y-%m-%d')
    else:
        min_date = df[date_col].min()

    full_period = relativedelta(df.date_col.max(), min_date)

    days = full_period.days
    months = full_period.months
    years = full_period.years

    TWO_THIRDS_YEAR_DAYS = 243
    TWO_THIRDS_YEAR_MONTHS = 8

    total_period_inputs = {'months': {'years': lambda x: x * 12,
                                     'days':  lambda x: 1 if x > 20 else 0,
                                     'months': lambda x: x},
                            'days': {'years': lambda x: x * 365,
                                     'days':  lambda x: x,
                                     'months': lambda x: x * 30},
                            'years': {'years': lambda x: x,
                                     'days':  lambda x: 1 if x >= TWO_THIRDS_YEAR_DAYS else 0,
                                     'months': lambda x: 1 if x >= TWO_THIRDS_YEAR_MONTHS else 0}
                            }
    total_periods = total_period_inputs[increment]['years'](days) + \
                   total_period_inputs[increment]['months'](days) + \
                   total_period_inputs[increment]['days'](days)

    periods_included = train_period_size
    
    tt_sets = []
    set_num = 0
    set_dates = pd.DataFrame(columns = ("set_num", "period", "train_start_date", 
                                        "train_end_date", "test_start_date", 
                                        "test_end_date"))
    
    while periods_included < total_periods:
        
        print(f"original train period length: {train_period_size}")

        train, test, date_ref = time_series_split(df, date_col=date_col, 
            train_size=train_period_size, test_size=test_period_size, 
            increment=increment, specify_start=specify_start)
        
        print(f"train: {train.shape}, test: {test.shape}")

        tt_sets.append((train, test))
        train_period_size += test_period_size
        periods_included += test_period_size

        set_dates.loc[set_num] = list(date_ref)
        set_num += 1

    return (tt_sets, set_dates)




def select_top_dummies(train_df, tops_list, threshold, max_options = 10):
    set_distro_dummies = []
    counter = 1
    dummies_dict = {}

    for col in tops_list:
        col_sum = train_df[col].value_counts().sum()
        top = train_df[col].value_counts().nlargest(max_options)
        
        top_value = 0
        num_dummies = 0

        while ((top_value / col_sum) < threshold) & (num_dummies < max_options):
            top_value += top[num_dummies]
            num_dummies += 1

        keep_dummies = list(top.index)[:num_dummies]
        dummies_dict[col] = keep_dummies
        
    counter += 1
    set_distro_dummies.append(dummies_dict)

    return set_distro_dummies



def apply_tops(set_specific_dummies, var_dict, train_df, test_df):
    counter = 0
    for set_dict in set_specific_dummies:
        counter += 1
        for col, vals in set_dict.items():
            train_df.loc[~train_df[col].isin(vals), col] = 'Other'
            test_df.loc[~test_df[col].isin(vals), col] = 'Other'




def iza_process(train_df, test_df, var_dict, tops_threshold = 0.5, binary = None, geos = False):
    # for i, (train_df, test_df) in enumerate(dfs):
    #     print("Starting set {}...".format(i))
        
    drop_unwanted(train_df, var_dict['datetime'])
    drop_unwanted(test_df, var_dict['datetime'])
    
    if binary is not None:
        make_boolean(train_df, var_dict['binary'], true_val = binary)
        make_boolean(test_df, var_dict['binary'], true_val = binary)
        print("Binary columns successfully converted.")

    
    train_df = replace_dummies(train_df, var_dict['multi'])
    test_df = replace_dummies(test_df, var_dict['multi'])
    
    # print("Values in columns {} successfully converted to dummies".format(var_dict['multi']))

    tops = select_top_dummies(train_df, var_dict['tops'], threshold = tops_threshold, max_options = 10)
    apply_tops(tops, var_dict, train_df, test_df)
    

    train_df = pd.get_dummies(train_df, columns = var_dict['tops'], dummy_na = True)
    test_df = pd.get_dummies(test_df, columns = var_dict['tops'], dummy_na = True)


    # print("Top values in columns {} successfully converted to dummies".format(var_dict['tops']))


    if geos:
        geo_tops = select_top_dummies(train_df, var_dict['geo'], threshold = tops_threshold, max_options = 5)
        apply_tops(geo_tops, var_dict, train_df, test_df)

        train_df = pd.get_dummies(train_df, columns = var_dict['geo'], dummy_na = True)
        test_df = pd.get_dummies(test_df, columns = var_dict['geo'], dummy_na = True)
        
        # print("Values in columns {} successfully converted to dummies".format(var_dict['geo']))
    print("Converted non-binary, non-numeric columns to dummies.")

    
    for col in var_dict['numeric']:
        basic_fill_vals(train_df, col_name = col, test_df = test_df, method = 'mean')
        train_df.loc[:, col] = normalize(pd.DataFrame(train_df[col]), axis = 0)
        test_df.loc[:, col] = normalize(pd.DataFrame(test_df[col]), axis = 0)
    print("Filled missing values and normalizied values in numeric columns.")

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    extra_train = train_cols - test_cols
    extra_test = test_cols - train_cols

    if len(extra_train) > 0:
        for col in extra_train:
            test_df.loc[:, col] = 0

    if len(extra_test) > 0:
        for col in extra_test:
            train_df.loc[:, col] = 0

    print("Moving to next set!")
    return (train_df, test_df)







