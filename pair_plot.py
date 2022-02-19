import sys
import re
import pandas as pd
import numpy as np
from csv_reader import read_csv
import matplotlib.pyplot as plt
import seaborn as sns


if (len(sys.argv) > 1):
    r = re.compile(".*.csv")
    args = list(filter(r.match, sys.argv))
    if (args):
        df = read_csv(args[0], index_col="Index")
    else:
        print("Bad file extension")
        sys.exit(1)

    sns.set_theme()
    sns.pairplot(df, hue="Hogwarts House", diag_kind='hist', plot_kws={'alpha': .5}, diag_kws={'alpha': .5}, corner=True)

    # print('"Arithmancy" and "Care of Magical Creatures" have a homogeneous score distribution between all four houses')
    if ('-s' in sys.argv or '--save' in sys.argv):
        print('Generating pairplot.png...')
        plt.savefig("pairplot")
        print('Pair plot saved to pairplot.png')
    else:
        plt.show()
    print("Feature chosen: Astronomy, Herbology, Ancient Runes, and Charms")
else:
    print("Wrong number of arguments")
    sys.exit(1)
