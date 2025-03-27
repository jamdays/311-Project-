import pandas as pd
import numpy as np
from scripts import clean_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("./data/clean_data.csv")
X = df
df = pd.read_csv("./data/cleaned_data_combined_modified.csv")
T = df["Label"]

np.random.seed(43)
X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.2, random_state=42)
X_train, X_valid, T_train, T_valid = train_test_split(X_train, T_train, test_size=0.2, random_state=42)

max_depths = [20, 30, 40, 50, 70, 100]
min_samples_split = [2, 4, 8, 16, 32, 64]
num_trees = [10, 50, 75, 100]

def build_all_models(max_depths,
                     min_samples_split,
                        X_train=X_train,
                        T_train=T_train,
                        X_valid=X_valid,
                        T_valid=T_valid):
    out = {}
    for d in max_depths:
        for s in min_samples_split:
            for num in num_trees:
                out[(d, s, num)] = {}
                forest = RandomForestClassifier(n_estimators=num, random_state=42, max_depth=d, min_samples_split=s)
                forest.fit(X_train, T_train)
                out[(d, s, num)]['val'] = forest.score(X_valid, T_valid)
                out[(d, s, num)]['train'] = forest.score(X_train, T_train)
    return out

res = build_all_models(max_depths=max_depths,min_samples_split=min_samples_split, X_train=X_train,T_train=T_train,X_valid=X_valid,T_valid=T_valid) # run build_all_models on all hyperparams
optimal=(-1,-1)
best_score=0
for d, s, num in res:
    if res[(d, s, num)]['val'] > best_score:
        best_score = res[(d, s, num)]['val']
        optimal = (d, s, num)
print("Best parameters: {}".format(optimal))
print("Best score: {}".format(best_score))

forest = RandomForestClassifier(n_estimators=optimal[2], random_state=42, max_depth=optimal[0], min_samples_split=optimal[1])
forest.fit(X_train, T_train)

print(forest.score(X_train, T_train))
print(forest.score(X_valid, T_valid))
print(forest.score(X_test, T_test))


def export_forest_structure(forest, feature_names):
    """
    Exports the structure of a random forest into a list of tree structures.
    """
    forest_structure = []
    for tree in forest.estimators_:
        tree_structure = export_tree_structure(tree, feature_names)
        forest_structure.append(tree_structure)
    return forest_structure

def export_tree_structure(decision_tree, feature_names):
    """
    Exports the structure of a decision tree into a dictionary format.
    """
    tree = decision_tree.tree_
    tree_structure = []

    for i in range(tree.node_count):
        node = {
            "node_id": i,
            "feature": feature_names[tree.feature[i]] if tree.feature[i] != -2 else "leaf",
            "threshold": float(tree.threshold[i]) if tree.feature[i] != -2 else None,
            "left_child": int(tree.children_left[i]) if tree.children_left[i] != -1 else None,
            "right_child": int(tree.children_right[i]) if tree.children_right[i] != -1 else None,
            "value": tree.value[i].tolist() if tree.feature[i] == -2 else None,
        }
        tree_structure.append(node)

    return tree_structure

feature_names = X_train.columns.tolist()
forest_structure = export_forest_structure(forest, feature_names)

# Convert the results dictionary into a DataFrame for easier plotting
results = []
for (d, s, num), scores in res.items():
    results.append({
        "max_depth": d,
        "min_samples_split": s,
        "num_trees": num,
        "val_score": scores['val'],
        "train_score": scores['train']
    })

results_df = pd.DataFrame(results)

# Plot validation scores for different max_depth and min_samples_split
plt.figure(figsize=(12, 8))
sns.lineplot(
    data=results_df,
    x="num_trees",
    y="val_score",
    hue="max_depth",
    style="min_samples_split",
    markers=True,
    palette="tab10"
)
plt.title("Validation Scores for Different Hyperparameters")
plt.xlabel("Number of Trees")
plt.ylabel("Validation Score")
plt.legend(title="Max Depth / Min Samples Split")
plt.grid(True)
plt.show()

with open("alt_forest_structure.json", "w") as f:
    json.dump(forest_structure, f, indent=4)
