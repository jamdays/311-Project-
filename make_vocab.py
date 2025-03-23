import pandas as pd
import numpy as np
from scripts import clean_data

df = pd.read_csv('./data/cleaned_data_combined_modified.csv')
X, T = clean_data(df)