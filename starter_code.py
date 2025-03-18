from scripts import clean_data
import pandas as pd

# Load the data
df = pd.read_csv("data/cleaned_data_combined_modified.csv")
X, T = clean_data(df)
