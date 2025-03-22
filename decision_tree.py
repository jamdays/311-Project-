from scripts import clean_data
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from graphviz import Source
from IPython.display import display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import json



np.random.seed(42)

# Load the data
df = pd.read_csv("./data/clean_data.csv")
X = df
T = pd.read_csv("./data/cleaned_data_combined_modified.csv")["Label"]

# Split the data into training and testing sets
X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.2, random_state=42)

# Split the data into training and validation sets
X_train, X_valid, T_train, T_valid = train_test_split(X_train, T_train, test_size=0.2, random_state=42)


# Optimal hyperparameters
# model = DecisionTreeClassifier()
# model.fit(X_train, T_train)

def build_all_models(max_depths,
                     min_samples_split,
                     criterion,
                     X_train=X_train,
                     t_train=T_train,
                     X_valid=X_valid,
                     t_valid=T_valid):
    """
    Parameters:
        `max_depths` - A list of values representing the max_depth values to be
                       try as hyperparameter values
        `min_samples_split` - An list of values representing the min_samples_split
                       values to try as hyperpareameter values
        `criterion` -  A string; either "entropy" or "gini"

    Returns a dictionary, `out`, whose keys are the the hyperparameter choices, and whose values are
    the training and validation accuracies (via the `score()` method).
    In other words, out[(max_depth, min_samples_split)]['val'] = validation score and
                    out[(max_depth, min_samples_split)]['train'] = training score
    For that combination of (max_depth, min_samples_split) hyperparameters.
    """
    out = {}

    for d in max_depths:
        for s in min_samples_split:
            out[(d, s)] = {}
            # Create a DecisionTreeClassifier based on the given hyperparameters and fit it to the data
            tree = DecisionTreeClassifier(criterion=criterion, min_samples_split=s, max_depth=d)
            tree.fit(X_train, T_train)
            out[(d, s)]['val'] = tree.score(X_valid, t_valid)
            out[(d, s)]['train'] = tree.score(X_train, t_train)
    return out


max_depths = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
min_samples_split = [2, 4, 8, 16, 32, 64, 128, 256, 512]



res = build_all_models(max_depths=max_depths,min_samples_split=min_samples_split,criterion="entropy",X_train=X_train,t_train=T_train,X_valid=X_valid,t_valid=T_valid) # run build_all_models on all hyperparams
optimal=(-1,-1)
best_score=0
for d, s in res:
    if res[(d, s)]['val'] > best_score:
        best_score = res[(d, s)]['val']
        optimal = (d, s)
print("Best parameters: {}".format(optimal))
print("Best score: {}".format(best_score))


model = DecisionTreeClassifier(criterion="entropy", min_samples_split=optimal[1], max_depth=optimal[0])
model.fit(X_train, T_train)
print(model.score(X_train, T_train))
print(model.score(X_valid, T_valid))
print(model.score(X_test, T_test))
dot_data = tree.export_graphviz(model)
graph = Source(dot_data)
graph.render("decision_tree", format="png", cleanup=True)  # Saves as decision_tree.png




# Extract data for plotting
max_depths = np.array(max_depths)
min_samples_split = np.array(min_samples_split)
validation_scores = np.array([[res[(d, s)]['val'] for s in min_samples_split] for d in max_depths])

# Create a meshgrid for the axes
X, Y = np.meshgrid(min_samples_split, max_depths)

# Plot the 3D surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, validation_scores, cmap='viridis', edgecolor='k')


# Add labels and title
ax.set_xlabel('Min Samples Split')
ax.set_ylabel('Max Depth')
ax.set_zlabel('Validation Accuracy')
ax.set_title('Validation Accuracy vs Max Depth and Min Samples Split')

# Add a color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.show()

# random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=optimal[0], min_samples_split=optimal[1])
# random_forest_model.fit(X_train, T_train)
# print(random_forest_model.score(X_train, T_train))
# print(random_forest_model.score(X_valid, T_valid))
# print(random_forest_model.score(X_test, T_test))

# Extract feature importances
print(model.tree_)