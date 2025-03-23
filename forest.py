from tree import Tree
import pandas as pd
import numpy as np
import json

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

if __name__ == "__main__":
    # Testing to see it if all works
    forest = RandomForest()
    forest.load_structure("forest_structure.json")
    X = pd.read_csv("data/clean_data.csv")
    T = pd.read_csv("data/cleaned_data_combined_modified.csv")["Label"]
    accuracy = 0
    for i in range(len(X)):
        if (forest.predict_one(X.iloc[i]) == T.iloc[i]):
            accuracy += 1
    accuracy = accuracy / len(X)
    print(accuracy)
