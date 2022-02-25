import re
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from csv_reader import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matrix_utils import standardize_df, standardize_df_test
from LR_model import LogisticRegression as LR


def log_reg_skl(df_train, df_test, features, target):

    # Get the mean & std by feature
    mean = df_train[features].mean()
    std = df_train[features].std()

    # Extract selected features and fill NAs with mean with the mean derived from each feature
    X_train = df_train[features].fillna(mean)
    y_train = df_train[target]
    X_test = df_test[features].fillna(mean)

    X_train = standardize_df(X_train)
    X_test = standardize_df_test(X_test, mean, std)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    pd.DataFrame(y_pred).to_csv(
        'houses.csv', header=[target], index_label='Index')
    y_true = pd.read_csv('houses_true.csv', sep=',', index_col="Index")

    # count nbr of true / false
    diff = (y_pred == y_true['Hogwarts House']).value_counts()
    print(diff)

    # Accuracy score
    score = accuracy_score(y_true=y_true, y_pred=y_pred) * 100
    print(f"accuracy_score: {score}%")


def log_reg(df_train, features, target):

    # Get the mean & std by feature
    mean = df_train[features].mean()
    std = df_train[features].std()

    # Extract selected features and fill NAs with mean with the mean derived from each feature
    X_train = df_train[features].fillna(mean)
    y_train = df_train[target]

    X_train = standardize_df(X_train)

    lr = LR()
    print('Logistic Regression in progress...')
    cost_history = lr.fit(X_train, y_train)
    print('Logistic Regression done !')
    lr.save_to_csv(mean, std)
    lr.plot_cost_history()
    plt.show()

    return cost_history

if (len(sys.argv) > 1):
    r = re.compile(".*.csv")
    args = list(filter(r.match, sys.argv))
    if (args):
        df_train = read_csv(args[0], index_col="Index")
        df_test = read_csv("datasets/dataset_test.csv", index_col="Index")

        features = ['Defense Against the Dark Arts',
                    'Herbology', 'Ancient Runes', 'Charms']
        target = 'Hogwarts House'

        cost_history = log_reg(df_train, features, target)

        # print()
        # log_reg_skl(df_train, df_test, features, target)
    else:
        print("Bad file extension")
        sys.exit(1)

else:
    print("Wrong number of arguments")
    sys.exit(1)
