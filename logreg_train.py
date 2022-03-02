import argparse
import re
import sys
import pandas as pd
from matplotlib import pyplot as plt
from csv_reader import read_csv
from matrix_utils import standardize_df, standardize_df_test
from LR_model import LogisticRegression as LR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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


def log_reg(df_train, df_test, features, target, solver):

    # Get the mean & std by feature
    mean = df_train[features].mean()
    std = df_train[features].std()

    # Extract selected features and fill NAs with mean with the mean derived from each feature
    X_train = df_train[features].fillna(mean)
    y_train = df_train[target]

    X_train = standardize_df(X_train)

    if solver == 'bgd':
        # Use Batch Gradient Descent
        lr = LR()
    elif solver == 'mgd':
        # Use Mini-Batch Gradient Descent
        lr = LR(nb_iter=3, solver='mgd')
    elif solver == 'sgd':
        # Use Stochastic Gradient Descent
        lr = LR(nb_iter=1, lr_ratio=0.2, solver='sgd')
    else:
        print(f'Bad solver: {solver}')
        sys.exit(1)

    print('Logistic Regression in progress...')
    lr.fit(X_train, y_train)
    print('Logistic Regression done !')

    lr.save_to_csv(features, mean, std)
    lr.plot_cost_history(mode='class')

    if df_test is not None:
        X_test = df_test[features].fillna(mean)
        X_test = standardize_df_test(X_test, mean, std)
        print("\nPredictions:\n============")
        y_pred = lr.predict(X_test)
        y_true = read_csv('houses_true.csv', index_col="Index")
        diff = (y_pred == y_true['Hogwarts House']).value_counts()
        print(diff)
        # Sklearn Accuracy score
        score = accuracy_score(y_true=y_true, y_pred=y_pred) * 100
        print(f"accuracy_score: {score}%")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Dataset for training as .csv')
    parser.add_argument('--solver', '-s', help='Algorithm to use in the optimization problem',
                        default='bgd', choices=['bgd', 'sgd', 'mgd'])
    parser.add_argument('--test', '-t', help='Dataset for testing as .csv')
    parser.add_argument(
        '--compare', '-c', help='Allows comparison with sklearn LR', action="store_true")

    args = parser.parse_args()

    match = re.search(".*.csv", args.dataset)
    if match:
        df_train = read_csv(match.group(0), index_col="Index")
    else:
        print(f"Bad file extension: {args.dataset}")
        sys.exit(1)

    features = ['Defense Against the Dark Arts',
                'Herbology', 'Ancient Runes', 'Charms']
    target = 'Hogwarts House'
    df_test = None

    if args.test:
        match = re.search(".*.csv", args.dataset)
        if match:
            df_test = read_csv(args.test, index_col="Index")
        else:
            print(f"Bad file extension: {args.dataset}")
            sys.exit(1)

    log_reg(df_train, df_test, features, target, args.solver)

    if args.compare and args.test:
        print("\nsklearn comparison:\n============")
        log_reg_skl(df_train, df_test, features, target)
