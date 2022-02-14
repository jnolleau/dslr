import pandas as pd
import matplotlib.pyplot as plt
from matrix_utils import get_numeric_col_names

df = pd.read_csv("datasets/dataset_train.csv")

courses = get_numeric_col_names(df)
courses.remove('Index')
nb_of_courses = len(courses)

plt.figure(figsize=(15,9))

for course in courses:
    plt.subplot((nb_of_courses // 3 + 1), 3, courses.index(course) + 1)
    plt.subplots_adjust(left=0.1, bottom=0.03, right=0.9, top=0.97, wspace=0.2, hspace=0.4)
    df.groupby(['Hogwarts House'])[course].plot(kind='hist', alpha=0.5)
    plt.legend()
    plt.title(course)

# # Output in full screen
# manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()
plt.show()
