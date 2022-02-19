import sys
import re
import matplotlib.pyplot as plt
import pandas as pd
from matrix_utils import get_numeric_col_names
from csv_reader import read_csv

def scatter_plot_selected(df, courses, target):
    # get all numeric columns (features)
    nb_of_courses = len(courses)
    
    # create a new column assigning an int to each house
    houses = df[target].unique().tolist()
    house_labels = {house : houses.index(house) for house in houses}
    df['house_label'] = df[target].replace(house_labels)

    # create the scatter plot
    for idx1 in range(nb_of_courses - 1):
        plt.figure(figsize=(12, 8))
        for idx2 in range(idx1, nb_of_courses):
            if (courses[idx1] != courses[idx2]):
                plt.scatter(
                    df[courses[idx2]], df[courses[idx1]], c=df['house_label'], alpha=0.5)
                plt.xlabel(courses[idx2])
                plt.ylabel(courses[idx1])
                plt.suptitle(f"Correlation between {courses[idx1]} and {courses[idx2]}")

def scatter_plot_all(df, target='Hogwarts House'):

    # get all numeric columns (features)
    courses = get_numeric_col_names(df)
    nb_of_courses = len(courses)
    
    # create a new column assigning an int to each house
    houses = df[target].unique().tolist()
    house_labels = {house : houses.index(house) for house in houses}
    df['house_label'] = df[target].replace(house_labels)

    # plt.figure(figsize=(12, 8))
    # for i in range(nb_of_courses):
    #     plt.subplot(5, 3, i + 1)
    #     plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95,
    #                         top=0.95, wspace=.3, hspace=0.6)
    #     plt.scatter(df[courses[i]], df.index, c=df['house_label'], s=5, alpha=0.5)
    #     plt.title(courses[i])

    # create the scatter plot
    for idx1 in range(nb_of_courses - 1):
        plt.figure(figsize=(12, 8))
        for idx2 in range(idx1, nb_of_courses):
            if (courses[idx1] != courses[idx2]):
                plt.subplot(4, 3, idx2 - idx1)
                plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                                    top=0.9, wspace=.3, hspace=0.6)
                plt.scatter(
                    df[courses[idx2]], df[courses[idx1]], c=df['house_label'], s=1, alpha=0.5)
                plt.xlabel(courses[idx2])
                plt.suptitle(courses[idx1])


if (len(sys.argv) > 1):
    r = re.compile(".*.csv")
    args = list(filter(r.match, sys.argv))
    if (args):
        df = read_csv(args[0], index_col="Index")
    else:
        print("Bad file extension")
        sys.exit(1)
    
    if ('-a' in sys.argv or '--all' in sys.argv):
        scatter_plot_all(df, target='Hogwarts House')
    else:
        selected = ['Astronomy', 'Defense Against the Dark Arts']
        scatter_plot_selected(df, courses=selected, target='Hogwarts House')
    print(f'{selected[0]} and {selected[1]} are the two similar features.')
    plt.show()

else:
    print("Wrong number of arguments")
    sys.exit(1)
