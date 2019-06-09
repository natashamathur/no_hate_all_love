import re
import pandas as pd
from collections import defaultdict



def summarize_df(df):
    """
    Note: Previously written function.
        
    Summarize columns in given pandas Dataframe including count of values, \
    number of unique values, number of null values, most common value, 
    frequency of most common value, and data type.
    """
    type_dict = defaultdict(list)
    geos = ["city", "state", "county", "country", "zip", "zipcode", 
            "latitude", "longitude"]
    geos = "|".join(geos)
    summary = pd.DataFrame(columns = ["col_name", "num_values", "num_nulls", 
                                      "unique_values",  "data_type", "col_type", 
                                      "most_common", "prevalence"])
    
    for col in df.columns:
        num_values = df[col].value_counts().sum()
        uniques = len(df[col].unique())
        nulls = df[col].isnull().sum()
        most_common = list(df[col].mode())[0]
        mode_count = (df[col].value_counts().max() / num_values) * 100
        dtype = df[col].dtype

        # automatically categorize based on data type for future explorations
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
    """
    Note: Previously written function.
    Recategorizes column type in col_dict output by summarize_df() function.
    
    Inputs:
        col: column to recategorize
        new_category: correct data type "category" for column
        col_dict: dictionary returned by output by summarize_df() function
        
    Output: updated col_dict dictionary
    """
    for category, cols_list in col_dict.items():
        if col in cols_list:
            col_dict[category] = [column for column in cols_list if column != col]
    col_dict[new_category].append(col)
    return col_dict


def individual_distribution(df, col_name, hist=False, ax=None, dropna=False):
    """
    Visualize distribution of values in a single column in a dataframe.
    
    Inputs:
        df: a pandas Dataframe
        col_name: name of column with values to be visualized
        hist: whether to layer histogram over Seaborn distplot
        ax: whther to add plot to an existing Matplotlib Axis
        dropna: whether to ignore NA values in data. Default is False.
    
    Output: Matplotlib Axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    if dropna:
        col_dist = sns.distplot(df[col_name].dropna(), hist=hist, ax=ax)
    else:
        col_dist = sns.distplot(df[col_name], hist=hist, ax=ax)
    renamed_x = col_dist.get_xlabel()
    renamed_x = renamed_x.replace('_', ' ').upper()
    col_dist.set_ylabel('FREQUENCY')
    col_dist.set_xlabel(renamed_x)
    col_dist.set_title(renamed_x + " DISTRIBUTION")
    return ax
