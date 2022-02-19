import math
import re
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import read_csv
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils import compute_class_weight
from matrix_utils import standardize_df


def log_reg_skl(df_train, df_test):
    features = ['Defense Against the Dark Arts',
                'Herbology', 'Ancient Runes', 'Charms']
    target = 'Hogwarts House'

    # Get the mean by feature
    mean = df_train[features].mean()

    # EXtract selected features and fill NAs with mean with the mean derived from each feature
    X_train = df_train[features].fillna(mean)
    y_train = df_train[target]
    X_test = df_test[features].fillna(mean)
    y_test = df_test[target]

    # Fill NAs within features with the median derived from each group
    # ==> Cannot be done because we don't have data for y_test
    # transformed_train = df_train.groupby(target).transform(lambda x: x.fillna(x.median()))
    # transformed_test = df_test.groupby(target).transform(lambda x: x.fillna(x.median()))

    # # Fill NAs for each raw according to k-NN algo
    # ==> sklearn as a fucntion for that
    # model = make_pipeline(KNNImputer(), SGDClassifier())
    # params = {
    #     'knnimputer__n_neighbors' : [1, 2, 3, 4]
    # }
    # grid = GridSearchCV(model, param_grid=params, cv=5)
    # grid.fit(X_train, y_train)
    # imputer = KNNImputer(n_neighbors=3)
    # X = imputer.fit_transform(X_train)
    # print(X[201:203])
    # print(df_train.groupby(target)['Astronomy'].median())
    # print("X_train['Astronomy'][213] = ", X_train['Astronomy'][213])

    # # For debug: display median for columns
    # for column in features:
    #     group = df_train.groupby(target)[column]
    #     median = group.median()
    #     print(median)

    # # For debug: display mean for columns
    # for column in features:
    #     median = df_train[column].mean()
    #     print(f"{column:15} {median:.9}")

    # plt.figure()
    # sns.boxplot(data=df_train['Defense Against the Dark Arts'])
    # plt.show()

    X_train = standardize_df(X_train)
    X_test = standardize_df(X_test)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print(clf.coef_)
    print(clf.n_iter_)
    y_pred = clf.predict(X_test)
    pd.DataFrame(y_pred).to_csv(
        'houses.csv', header=[target], index_label='Index')
    y_true = pd.read_csv('houses_true.csv', sep=',', index_col="Index")
    score = accuracy_score(y_true=y_true, y_pred=y_pred) * 100
    print(f"accuracy_score: {score}%")


# 1. model F
def model(X, W):
    return X.dot(W)


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

# def log_loss(y, A):
#   return 1/len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

# 3. Gradient descent


def grad(X, y, H0):
    m = len(y)
    dW = 1/m * np.dot(X.T, H0 - y)
    db = 1/m * np.sum(H0 - y)
    return dW, db


def gradient_descent(X, y, H0, W, b, learningRate):
    dW, db = grad(X, y, H0)
    W = W - learningRate * dW
    b = b - learningRate * db
    return W, b


def log_reg(df_train, df_test):
    features = ['Defense Against the Dark Arts',
                'Herbology', 'Ancient Runes', 'Charms']
    target = 'Hogwarts House'

    # Get the mean by feature
    mean = df_train[features].mean()

    # EXtract selected features and fill NAs with mean with the mean derived from each feature
    X_train = df_train[features].fillna(mean)
    y_train = df_train[target]
    X_test = df_test[features].fillna(mean)
    y_test = df_test[target]

    X_train = standardize_df(X_train)
    X_test = standardize_df(X_test)

    # Create a Series containing class index
    classes = y_train.unique()
    nb_of_class = classes.shape[0]
    house_labels = {house: i for house, i in zip(classes, range(nb_of_class))}
    y = y_train.replace(house_labels)
    y = y.values.reshape(y.shape[0], 1)
    W_all = np.zeros((nb_of_class, X_train.shape[1]))
    epoch = 500
    cost_history = []
    for curr_class in range(nb_of_class):
        W, b = initialisation(X_train)
        y_class = y
        y_class[y_class != curr_class] = 0
        y_class[y_class == curr_class] = 1
        for i in range(0, epoch):
            H0 = H0_calculation(X_train, W, b)
            W, b = gradient_descent(X_train, y_class, H0, W, b, learningRate=0.05)
            cost_history.append(cost_function(y_class, H0))
        W_all[curr_class] = W.T
    print(W_all)
    return cost_history


if (len(sys.argv) > 1):
    r = re.compile(".*.csv")
    args = list(filter(r.match, sys.argv))
    if (args):
        df_train = read_csv(args[0], index_col="Index")
        df_test = read_csv("datasets/dataset_test.csv", index_col="Index")
        # cost_history = log_reg(df_train, df_test)
        # plt.figure(figsize=(9, 6))
        # plt.plot(cost_history)
        # plt.xlabel('n_iteration')
        # plt.ylabel('Log_loss')
        # plt.title('Evolution des erreurs')
        # plt.show()

        # print()
        log_reg_skl(df_train, df_test)
    else:
        print("Bad file extension")
        sys.exit(1)

else:
    print("Wrong number of arguments")
    sys.exit(1)
