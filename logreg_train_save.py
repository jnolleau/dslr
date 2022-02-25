import re
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matrix_utils import standardize_df, standardize_df_test


def log_reg_skl(df_train, df_test, features, target):

    # Get the mean & std by feature
    mean = df_train[features].mean()
    std = df_train[features].std()

    # EXtract selected features and fill NAs with mean with the mean derived from each feature
    X_train = df_train[features].fillna(mean)
    y_train = df_train[target]
    X_test = df_test[features].fillna(mean)

    X_train = standardize_df(X_train)
    X_test = standardize_df_test(X_test, mean, std)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    # print('weights', clf.coef_)
    # print('bias', clf.intercept_)
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


# 1. model F

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)


def H0_calculation(X, W, b):
    Z = X.dot(W) + b
    H0 = sigmoid(Z)
    return H0

# 2. Fonction cout ( == log_loss)


def cost_function(y, H0):
    m = len(y)
    return -1/m * np.sum(y * np.log(H0) + (1 - y) * np.log(1 - H0))


# 3. Gradient descent

def grad(X, y, H0):
    m = len(y)
    dW = 1/m * np.dot(X.T, H0 - y)
    db = 1/m * np.sum(H0 - y)
    return dW, db


def gradient_descent(X, y, W, b, learningRate):
    n_iteration = 150
    cost_history = np.zeros((n_iteration, 1))
    for i in range(n_iteration):
        H0 = H0_calculation(X, W, b)
        dW, db = grad(X, y, H0)
        W = W - learningRate * dW
        b = b - learningRate * db
        cost_history[i] = cost_function(y, H0)
    return W, b, cost_history

# Prediction


def predict(X, W_all, b_all, classes):
    H0 = pd.DataFrame()
    for W, b in zip(W_all, b_all.T):
        # W = W.reshape(W.shape[0], 1)
        H0_class = H0_calculation(X, W.T, b)
        H0 = pd.concat([H0, H0_class], axis=1)
    H0.columns = classes
    y_pred = H0.idxmax(axis=1)
    pd.DataFrame(y_pred).to_csv(
        'houses.csv', header=['House'], index_label='Index')

    y_true = pd.read_csv('houses_true.csv', sep=',', index_col="Index")
    score = accuracy_score(y_true=y_true, y_pred=y_pred) * 100
    diff = (y_pred == y_true['Hogwarts House']).value_counts()
    print(diff)
    return y_pred, score


def log_reg(df_train, df_test, features, target):

    # Get the mean & std by feature
    mean = df_train[features].mean()
    std = df_train[features].std()

    # EXtract selected features and fill NAs with mean with the mean derived from each feature
    X_train = df_train[features].fillna(mean)
    y_train = df_train[target]
    X_test = df_test[features].fillna(mean)
    y_test = df_test[target]

    X_train = standardize_df(X_train)
    X_test = standardize_df_test(X_test, mean, std)

    # Create a Series containing class index
    classes = y_train.unique()
    nb_of_class = classes.shape[0]
    house_labels = {house: i for house, i in zip(classes, range(nb_of_class))}
    print(house_labels)
    y_train = y_train.replace(house_labels)
    y_train = y_train.values.reshape(y_train.shape[0], 1)
    W = np.zeros((nb_of_class, X_train.shape[1]))
    b = np.zeros((1, nb_of_class))
    cost_history = np.zeros((1, 1))
    # Wb_dict = {f'W{i}': [] for i in range(nb_of_class)}
    # Wb_dict['b'] = []

    for curr_class in range(nb_of_class):
        W_class, b_class = initialisation(X_train)
        y = np.where(y_train == curr_class, 1, 0)
        W_class, b_class, cost_history_class = gradient_descent(
            X_train, y, W_class, b_class, learningRate=0.1)
        W[curr_class] = W_class.T
        b[0:, curr_class] = b_class
        cost_history = np.concatenate(
            (cost_history, cost_history_class), axis=0)
        # Wb_dict[f'W{curr_class}'].append(W_class.T)
        # Wb_dict['b'].append(b_class[0])

    y_pred, score = predict(X_test, W, b, classes)
    print(y_pred)
    print(f"accuracy_score: {score}%")

    return W, b, cost_history


if (len(sys.argv) > 1):
    r = re.compile(".*.csv")
    args = list(filter(r.match, sys.argv))
    if (args):
        df_train = read_csv(args[0], index_col="Index")
        df_test = read_csv("datasets/dataset_test.csv", index_col="Index")

        features = ['Defense Against the Dark Arts',
                    'Herbology', 'Ancient Runes', 'Charms']
        target = 'Hogwarts House'

        W, b, cost_history = log_reg(df_train, df_test, features, target)

        # # Create a dataframe with W and b to store it nicely
        # Wb_dict = {f'W{i}': W[i] for i in range(W.shape[1])}
        # Wb_dict['b'] = b[0]
        # weights = pd.DataFrame(Wb_dict, index=df_train[target].unique())
        # weights.to_csv('weights.csv')
        # print(weights)
        # print(weights.loc['W0'])
        # plt.figure(figsize=(9, 6))
        # plt.plot(cost_history)
        # plt.xlabel('n_iteration')
        # plt.ylabel('Log_loss')
        # plt.title('Evolution des erreurs')
        # plt.show()

        # print()
        # log_reg_skl(df_train, df_test, features, target)
    else:
        print("Bad file extension")
        sys.exit(1)

else:
    print("Wrong number of arguments")
    sys.exit(1)
