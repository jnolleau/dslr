import sys
import re
import numpy as np
import pandas as pd
from math_utils import *
from matrix_utils import get_numeric_col_names
from csv_reader import read_csv


def describe(df, ignore_index=False):
    numeric_col_names = get_numeric_col_names(df)
    if (ignore_index == True):
        numeric_col_names = numeric_col_names[1:]
    stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    desc_dict = {k: [] for k in stats}
    for item in numeric_col_names:
        mean_count = mean_column(df[item].dropna())
        std = std_column(df[item].dropna())
        min_max = min_max_column(df[item].dropna())
        per = percentiles_column(df[item].dropna())
        desc_dict[stats[0]].append(mean_count[1])
        desc_dict[stats[1]].append(mean_count[0])
        desc_dict[stats[2]].append(std[0])
        desc_dict[stats[3]].append(min_max[0])
        desc_dict[stats[4]].append(per[0])
        desc_dict[stats[5]].append(per[1])
        desc_dict[stats[6]].append(per[2])
        desc_dict[stats[7]].append(min_max[1])
    df1 = pd.DataFrame.from_dict(
        desc_dict, orient='index', columns=numeric_col_names)
    return df1

# def describe(df, ignore_index=False):
#     numeric_col_names = get_numeric_col_names(df)
#     if (ignore_index == True):
#         numeric_col_names = numeric_col_names[1:]
#     desc_dict = {
#         'stats.': ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
#     }
#     for item in numeric_col_names:
#         mean_count = mean_column(df[item].dropna())
#         std = std_column(df[item].dropna())
#         min_max = min_max_column(df[item].dropna())
#         per = percentiles_column(df[item].dropna())
#         desc_dict[item] = [mean_count[1], mean_count[0], std[0], min_max[0], per[0], per[1], per[2], min_max[1]]
#     df1 = pd.DataFrame.from_dict(desc_dict).set_index('stats.')
#     return df1


if (len(sys.argv) > 1):
    r = re.compile(".*.csv")
    args = list(filter(r.match, sys.argv))
    if (args):
        df = read_csv(args[0])
    else:
        print("Bad file extension")
        sys.exit(1)
    df1 = describe(df, ignore_index=True)
    if ('-f' in sys.argv or '--full' in sys.argv):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df1)
    if ('-s' in sys.argv or '--save' in sys.argv):
        df1.to_csv('stats.csv')
    else:
        print(df1)
else:
    print("Wrong number of arguments")
    sys.exit(1)
