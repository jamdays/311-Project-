import numpy as np
import pandas as pd
from tree import Tree
from forest import RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


model = RandomForest()
model.load_structure("forest_structure.json")

df = pd.read_csv("./data/clean_data.csv")
X = df
df = pd.read_csv("./data/cleaned_data_combined_modified.csv")
T = df["Label"]

np.random.seed(43)
X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.2, random_state=42)
X_train, X_valid, T_train, T_valid = train_test_split(X_train, T_train, test_size=0.2, random_state=42)


def plot_confusion_matrix(X, t, lr=model, group = "Everyone"):
    """
    Use the sklearn model "lr" to make predictions for the data "X",
    then compare the prediction with the target "t" to plot the confusion matrix.

    Moreover, this function prints the accuracy, precision and recall
    """
    cm = confusion_matrix(t, lr.predict_all(X))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pizza", "Shawarma", "Sushi"])
    disp.plot()
    plt.title(f"Confusion Matrix for {group}")

    plt.show()

plot_confusion_matrix(X_train, T_train, group="Training")
plot_confusion_matrix(X_valid, T_valid, group="Validation")