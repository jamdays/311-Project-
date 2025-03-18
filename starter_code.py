from scripts import clean_data
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from graphviz import Source
from IPython.display import display


# Load the data
df = pd.read_csv("./data/clean_data.csv")
X = df
T = pd.read_csv("./data/cleaned_data_combined_modified.csv")["Label"]

X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, T_train)
print(model.score(X_test, T_test))
dot_data = tree.export_graphviz(model)
graph = Source(dot_data)
graph.render("decision_tree", format="png", cleanup=True)  # Saves as decision_tree.png
