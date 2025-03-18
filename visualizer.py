from graphviz import Source

# Load and visualize the tree.dot file
dot_file_path = "tree.dot"
graph = Source.from_file(dot_file_path)
graph.render("tree", format="png", cleanup=True)  # Saves as tree.png
graph.view()  # Opens the visualization