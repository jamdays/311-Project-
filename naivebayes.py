import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from scripts import *
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data_combined_modified.csv")
    X, T = clean_data(df)
    cols = ['Friends', 'Strangers', 'Siblings', 'Family', 'Teachers', 'price',
       'num_ingredients', 'complexity', 'Weekend dinner', 'Weekend lunch',
       'Late night snack', 'At a party', 'Week Day lunch', 'Week day dinner',
       'mild', 'None', 'hot', 'medium']
    # sets = set(powerset(cols)) - set(cols)
    # single_sets = [cols[5:], cols[:5] + cols[6:], cols[:6] + cols[7:],
    #         cols[:7] + cols[8:], cols[:8] + cols[14:], cols[:14]]
    # missing_one_sets = [cols[:5], [cols[5]], [cols[6]], [cols[7]], cols[8:14], cols[14:]]
    # missing_two_sets = [cols[:6], cols[:5] + [cols[6]], cols[:5] + [cols[7]],
    #                     cols[:5] + cols[8:14], cols[:5] + cols[14:],
    #                     cols[5:7], [cols[5]] + [cols[7]], [cols[5]] + cols[8:14]
    #                     , [cols[5]] + cols[14:], cols[6:8], [cols[6]] +
    #                     cols[8:14], [cols[6]] + cols[14:], cols[7:14],
    #                     [cols[7]] + cols[14:], cols[8:]]
    subset_included = []
    accuracy_score_w_fit = []
    accuracy_score_wo_fit = []
    sets = [[]]
    for s in sets:
        Xc = X.copy()
        for col in s:
            Xc.drop(columns=[col], inplace=True)
        x_train, x_test_valid, t_train, t_test_valid = train_test_split(
            Xc, T, test_size=0.3, random_state=99)
        x_valid, x_test, t_valid, t_test = train_test_split(
            x_test_valid, t_test_valid, test_size=0.5, random_state=99)

        classifier = GaussianNB(priors=[1/3, 1/3, 1/3])
        classifier.fit(x_train, t_train)
        y_pred = classifier.predict(x_valid)
        accuracy_score_wo_fit.append(accuracy_score(t_valid, y_pred))
        if accuracy_score(t_valid, y_pred) > .78:
            print("no_fit")
            print(accuracy_score(t_valid, y_pred))
            print(set(cols) - set(s))

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_valid = sc.transform(x_valid)
        x_test = sc.transform(x_test)
        classifier = GaussianNB()
        classifier.fit(x_train, t_train)
        y_pred = classifier.predict(x_valid)

        accuracy_score_w_fit.append(accuracy_score(t_valid, y_pred))
        if accuracy_score(t_valid, y_pred) > .78:
            print("fit")
            print(accuracy_score(t_valid, y_pred))
            print(set(cols) - set(s))
        subset_included.append(set(cols) - set(s))

    print(max(accuracy_score_w_fit))
    print(max(accuracy_score_wo_fit))
    if max(accuracy_score_w_fit) > max(accuracy_score_wo_fit):
        print(subset_included[accuracy_score_w_fit.index(max(accuracy_score_w_fit))])
    else:
        print(subset_included[accuracy_score_wo_fit.index(max(accuracy_score_wo_fit))])

