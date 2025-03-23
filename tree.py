import json
import numpy as np
import pandas as pd

class Tree:
    def __init__(self):
        self.root = None
        self.tree_structure = None

    def load_structure(self, filename):
        with open(filename, 'r') as f:
            self.tree_structure = json.load(f)
    
    def set_structure(self, tree_structure):
        self.tree_structure = tree_structure
    
    def predict_one(self, input):
        curr_node = self.tree_structure[0]
        while curr_node["feature"] != "leaf":
            if input[curr_node["feature"]] <= curr_node["threshold"]:
                curr_node = self.tree_structure[curr_node["left_child"]]
            else:
                curr_node = self.tree_structure[curr_node["right_child"]]
        prediction = np.argmax(curr_node["value"][0])
        prediction = ["Pizza", "Shawarma", "Sushi"][prediction]
        return prediction
    
    def predict_all(self, df):
        predictions = []
        for index, row in df.iterrows():
            predictions.append(self.predict_one(row))
        return predictions

if __name__ == "__main__":
    # Test to see if it runs
    tree = Tree()
    tree.load_structure("decision_tree_structure.json")
    X = pd.read_csv("data/clean_data.csv")
    T = pd.read_csv("data/cleaned_data_combined_modified.csv")["Label"]
    accuracy = 0
    predictions = tree.predict_all(X)
    for i in range(len(X)):
        if (predictions[i] == T.iloc[i]):
            accuracy += 1
    accuracy = accuracy / len(X)
    print(accuracy)