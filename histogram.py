import sys
import re
import matplotlib.pyplot as plt
from csv_reader import read_csv
from matrix_utils import get_numeric_col_names


def histogram(df):

    # get all numeric columns (features)
    courses = get_numeric_col_names(df)
    nb_of_courses = len(courses)

    plt.figure(figsize=(15, 9))
    # # Output in full screen
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()

    for course in courses:
        plt.subplot((nb_of_courses // 3 + 1), 3, courses.index(course) + 1)
        plt.subplots_adjust(left=0.1, bottom=0.03, right=0.9,
                            top=0.97, wspace=0.2, hspace=0.4)
        df.groupby(['Hogwarts House'])[course].plot(kind='hist', alpha=0.5)
        plt.legend()
        plt.title(course)


if (len(sys.argv) > 1):
    r = re.compile(".*.csv")
    args = list(filter(r.match, sys.argv))
    if (args):
        df = read_csv(args[0], index_col="Index")
    else:
        print("Bad file extension")
        sys.exit(1)

    histogram(df)
    print('"Arithmancy" and "Care of Magical Creatures" have a homogeneous score distribution between all four houses')
    plt.show()

else:
    print("Wrong number of arguments")
    sys.exit(1)
