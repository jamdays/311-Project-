import json
import numpy as np
import pandas as pd

def import_tree_structure(filename):
    with open(filename, 'r') as f:
        tree_structure = json.load(f)
    return tree_structure

def predict(tree_structure, input):
    curr_node = tree_structure[0]
    while curr_node["feature"] != "leaf":
        if input[curr_node["feature"]] <= curr_node["threshold"]:
            curr_node = tree_structure[curr_node["left_child"]]
        else:
            curr_node = tree_structure[curr_node["right_child"]]
    prediction = np.argmax(curr_node["value"][0])
    prediction = ["Pizza", "Shawarma", "Sushi"][prediction]
    return prediction

def predict_all(filename):
    with open(filename, 'r') as f:
        raw_data = pd.read_csv(f)
    
    return "Pizza"

tree_structure = import_tree_structure("decision_tree_structure.json")
X = pd.read_csv("data/clean_data.csv")
T = pd.read_csv("data/cleaned_data_combined_modified.csv")["Label"]
accuracy = 0
for i in range(len(X)):
    if (predict(tree_structure, X.iloc[i]) == T.iloc[i]):
        accuracy += 1
accuracy = accuracy / len(X)
print(accuracy)