import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("datasets/dataset_train.csv")

#Normalisation
def get_numeric_col_names(df):
    return df.select_dtypes(include=np.number).columns.tolist()

courses = get_numeric_col_names(df)
courses.remove('Index')
nb_of_courses = len(courses)

# corr = df.corr()
sns.set_theme()
sns.pairplot(df, hue="Hogwarts House", diag_kind='hist', plot_kws={'alpha': .5}, diag_kws={'alpha': .5}, corner=True)
plt.savefig("pairplot")
# plt.show()
