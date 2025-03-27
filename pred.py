import json
import numpy as np
import pandas as pd
from scripts import clean_data
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

class RandomForest:
    def __init__(self):
        self.trees = []
    
    def load_structure(self, filename):
        with open(filename, 'r') as f:
            forest_structure = json.load(f)
        for tree_structure in forest_structure:
            tree = Tree()
            tree.set_structure(tree_structure)
            self.trees.append(tree)
    
    def predict_one(self, input):
        predictions = [tree.predict_one(input) for tree in self.trees]
        unique, counts = np.unique(predictions, return_counts=True)
        return unique[np.argmax(counts)]

    def predict_all(self, df):
        predictions = []
        for index, row in df.iterrows():
            predictions.append(self.predict_one(row))
        return predictions

def predict_all(filename):
    forest = RandomForest()
    forest.load_structure("forest_structure.json")
    df = pd.read_csv(filename)
    X, T = clean_data(df, clean_type="no_words")
    return forest.predict_all(X)

if __name__ == "__main__":
    predictions = predict_all("data/cleaned_data_combined_modified.csv")
    print(predictions)