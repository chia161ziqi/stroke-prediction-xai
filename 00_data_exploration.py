# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
from utils import DataLoader

# %% Load data
data_loader = DataLoader()
data_loader.load_dataset()
data = data_loader.data

# %% Show head
print("\nDataset Shape:")
print(data.shape)

print("\nFirst 5 Rows:")
data.head()

# %% Show general statistics
data.info()

print("\nColumn Names:")
print(data.columns.tolist())

# %% Show histogram for all columns
columns = data.columns
for col in columns:
    print("col: ", col)
    data[col].hist()
    plt.show()

# %% Show preprocessed dataframe
data_loader.preprocess_data()
data_loader.data.head()

