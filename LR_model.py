import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from math_utils import sigmoid

class LogisticRegression:
    def __init__(self):
        self.W = None
        self.b = None
        self.classes = None
        self.features = None
        self.cost_history = np.zeros((1, 1))

    # 1. model F
    def __initialisation(self, nb_of_features):
        W = np.ones((nb_of_features, 1))
        b = np.ones(1)
        return (W, b)

    def __H0_calculation(self, X, W, b):
        Z = X.dot(W) + b
        H0 = sigmoid(Z)
        return H0

    # 2. Fonction cout (== log_loss)
    def __cost_function(self, y, H0):
        m = len(y)
        return -1/m * np.sum(y * np.log(H0) + (1 - y) * np.log(1 - H0))

    # 3. Gradient descent
    def __grad(self, X, y, H0):
        m = len(y)
        dW = 1/m * np.dot(X.T, H0 - y)
        db = 1/m * np.sum(H0 - y)
        return dW, db

    def __gradient_descent(self, X, y, W, b, nb_iter, learningRate):

        cost_history = np.zeros((nb_iter, 1))
        for i in range(nb_iter):
            H0 = self.__H0_calculation(X, W, b)
            dW, db = self.__grad(X, y, H0)
            W = W - learningRate * dW
            b = b - learningRate * db
            cost_history[i] = self.__cost_function(y, H0)
        return W, b, cost_history

    # 4. Regression
    def fit(self, X_train, y_train, nb_iter=150, learningRate=0.1):

        # Create a Series containing class index
        self.classes = y_train.unique()
        self.features = X_train.columns
        nb_of_class = self.classes.shape[0]
        house_labels = {house: i for house, i in zip(
            self.classes, range(nb_of_class))}

        y_train = y_train.replace(house_labels)
        y_train = y_train.values.reshape(y_train.shape[0], 1)
        self.W = np.zeros((nb_of_class, X_train.shape[1]))
        self.b = np.zeros((1, nb_of_class))

        for curr_class in range(nb_of_class):
            W_class, b_class = self.__initialisation(X_train.shape[1])
            y = np.where(y_train == curr_class, 1, 0)
            W_class, b_class, cost_history_class = self.__gradient_descent(
                X_train, y, W_class, b_class, nb_iter, learningRate)
            self.W[curr_class] = W_class.T
            self.b[0:, curr_class] = b_class
            self.cost_history = np.concatenate(
                (self.cost_history, cost_history_class), axis=0)

        return self.cost_history

    # 5. Prediction
    def predict(self, X, weights=None, intercept=None, classes=None):

        if(weights is not None):
            self.W = weights.to_numpy()
        if(intercept is not None):
            self.b = intercept

        H0 = None
        for W_class, b in zip(self.W, self.b.T):
            H0_class = self.__H0_calculation(X, W_class.T, b)
            H0 = pd.concat([H0, H0_class], axis=1)
        if classes is not None:
            self.classes = classes
        H0.columns = self.classes
        y_pred = H0.idxmax(axis=1)

        return y_pred

    def accuracy_score(y_true, y_pred):
        score = accuracy_score(y_true=y_true, y_pred=y_pred) * 100
        diff = (y_pred == y_true).value_counts()
        print(diff)
        return score

    def save_to_csv(self, mean, std):
        # Create a dataframe with W
        Wb_dict = {feature: self.W[i] for feature, i in zip(
            self.features, range(self.features.shape[0]))}

        # Add bias, mean and std to dataframe
        for feature, i in zip(self.features, range(self.features.shape[0])):
            Wb_dict[feature] = np.append(Wb_dict[feature], self.b[0, i])
            Wb_dict[feature] = np.append(Wb_dict[feature], mean.loc[feature])
            Wb_dict[feature] = np.append(Wb_dict[feature], std.loc[feature])

        col = np.append(self.classes, ['b', 'mean', 'std'])
        weights = pd.DataFrame.from_dict(Wb_dict, orient='index', columns=col)
        weights.index.name = 'weights'
        weights.to_csv('weights.csv')
        print(weights[self.classes])

    def plot_cost_history(self):
        plt.figure(figsize=(9, 6))
        plt.plot(self.cost_history)
        plt.xlabel('n_iteration')
        plt.ylabel('Log_loss')
        plt.title('Evolution des erreurs')
