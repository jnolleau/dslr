import argparse
import re
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from matrix_utils import standardize_df_test
from LR_model import LogisticRegression as LR
from csv_reader import read_csv


def predict(df_test, features, target, model, truth):

    # Get all the parameters from model
    b = model['b']
    mean = model['mean']
    std = model['std']
    model.drop(['b', 'mean', 'std'], axis=1, inplace=True)

    # Extract selected features and fill NAs with mean with the mean derived from each feature
    X_test = df_test[features].fillna(mean)
    X_test = standardize_df_test(X_test, mean, std)

    lr = LR()
    # Prediction
    y_pred = lr.predict(X_test, weights=model.values, intercept=b, classes=np.asarray(model.columns))
    y_true = read_csv(truth, index_col="Index")

    # count nbr of true / false
    diff = (y_pred == y_true[target]).value_counts()
    print(diff)
    
    # Sklearn Accuracy score
    score = accuracy_score(y_true=y_true, y_pred=y_pred) * 100
    print(f"accuracy_score: {score}%")

    # Export y_pred
    pd.DataFrame(y_pred).to_csv(
    'houses.csv', header=[target], index_label='Index')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Dataset for testings as .csv')
    parser.add_argument(
        '--truth', '-t', help='Filename containing true predictions', default='houses_true.csv')

    args = parser.parse_args()

    match = re.search(".*.csv", args.dataset)
    if match:
        df_test = read_csv(match.group(0), index_col="Index")
    else:
        print(f"Bad file extension: {args.dataset}")
        sys.exit(1)

    model = read_csv('weights.csv', index_col='weights')
    features = ['Defense Against the Dark Arts',
                'Herbology', 'Ancient Runes', 'Charms']
    target = 'Hogwarts House'

    predict(df_test, features, target, model, args.truth)
