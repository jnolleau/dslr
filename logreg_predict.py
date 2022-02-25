import re
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from matrix_utils import standardize_df_test
from LR_model import LogisticRegression as LR
from csv_reader import read_csv


def predict(df_test, features, target, model):

    # Get all the parameters from model
    b = model['b']
    mean = model['mean']
    std = model['std']
    model.drop(['b', 'mean', 'std'], axis=1, inplace=True)
    # print(model)

    # Extract selected features and fill NAs with mean with the mean derived from each feature
    X_test = df_test[features].fillna(mean)
    X_test = standardize_df_test(X_test, mean, std)

    lr = LR()
    # Prediction
    y_pred = lr.predict(X_test, weights=model, intercept=b, classes=model.columns)
    y_true = read_csv('houses_true.csv', index_col="Index")

    # count nbr of true / false
    diff = (y_pred == y_true['Hogwarts House']).value_counts()
    print(diff)
    
    # Sklearn Accuracy score
    score = accuracy_score(y_true=y_true, y_pred=y_pred) * 100
    print(f"accuracy_score: {score}%")

    # Export y_pred
    pd.DataFrame(y_pred).to_csv(
    'houses.csv', header=[target], index_label='Index')

if (len(sys.argv) > 1):
    r = re.compile(".*.csv")
    args = list(filter(r.match, sys.argv))
    if (args):
        df_test = read_csv(args[0], index_col="Index")
        model = read_csv('weights.csv', index_col='weights')

        features = ['Defense Against the Dark Arts',
                    'Herbology', 'Ancient Runes', 'Charms']
        target = 'Hogwarts House'

        predict(df_test, features, target, model)

    else:
        print("Bad file extension")
        sys.exit(1)

else:
    print("Wrong number of arguments")
    sys.exit(1)
