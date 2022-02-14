import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("datasets/dataset_train.csv")

def get_numeric_col_names(df):
    return df.select_dtypes(include=np.number).columns.tolist()


courses = get_numeric_col_names(df)
courses.remove('Index')
nb_of_courses = len(courses)

# Normalisation
normalized_df = df.copy()
# for course in courses:
#     normalized_df[course] = (normalized_df[course]-normalized_df[course].mean())/normalized_df[course].std()

# To create a new column assigning an int to each house
house_labels = {
    'Ravenclaw': 1,
    'Slytherin': 2,
    'Gryffindor': 3,
    'Hufflepuff': 4
}
normalized_df['house_label'] = normalized_df['Hogwarts House'].replace(
    house_labels)

for idx1 in range(nb_of_courses - 1):
    plt.figure(figsize=(12, 8))
    for idx2 in range(idx1, nb_of_courses):
        if (courses[idx1] != courses[idx2]):
            plt.subplot(4, 3, idx2 - idx1)
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                                top=0.9, wspace=.3, hspace=0.6)
            plt.scatter(
                normalized_df[courses[idx2]], normalized_df[courses[idx1]], c=normalized_df['house_label'], s=1, alpha=0.5)
            plt.xlabel(courses[idx2])
            plt.suptitle(courses[idx1])
plt.show()
