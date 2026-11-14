# %% Imports
from pathlib import Path
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
print(data.head())

# %% Show general statistics
data.info()

print("\nColumn Names:")
print(data.columns.tolist())

# %% Data Checks - Duplicate Records
print("\nDuplicate Records:")

# Fully duplicated rows
duplicate_rows = data.duplicated().sum()
print(f"\nDuplicate Rows: {duplicate_rows}")

# Duplicates excluding patient ID
duplicate_without_id = (
    data.drop(columns=["id"]).duplicated().sum()
)
print(f"Duplicate Records (excluding ID): {duplicate_without_id}")

# Duplicate patient IDs
duplicate_ids = data["id"].duplicated().sum()
print(f"Duplicate IDs: {duplicate_ids}")

# %% Data Checks - Missing Values
print("\nMissing Values:")

missing_counts = data.isnull().sum()
missing_percentages = data.isnull().mean() * 100

missing_summary = pd.DataFrame({
    "Missing Count": missing_counts,
    "Missing Percentage (%)": missing_percentages.round(2)
})

print(missing_summary)

# %% Stroke Class Counts
print("\nStroke Class Counts:")

stroke_counts = data["stroke"].value_counts().sort_index()
stroke_percentages = (
    data["stroke"].value_counts(normalize=True).sort_index() * 100
)

stroke_summary = pd.DataFrame({
    "Count": stroke_counts,
    "Percentage (%)": stroke_percentages.round(2)
})

print(stroke_summary)

# Class imbalance ratio
if 0 in stroke_counts.index and 1 in stroke_counts.index:
    if stroke_counts[1] > 0:
        imbalance_ratio = stroke_counts[0] / stroke_counts[1]
        print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
    else:
        print("\nClass Imbalance Ratio: Undefined (no stroke=1 records)")

# %% Age, glucose and BMI
print("\nAge, Glucose and BMI Statistics:")

numerical_cols = [
    "age",
    "avg_glucose_level",
    "bmi"
]

print("\nDescriptive Statistics:")
print(data[numerical_cols].describe().round(2))

print("\nMean, SD, Median, IQR and Range:")

for col in numerical_cols:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)

    mean = data[col].mean()
    std = data[col].std()
    median = data[col].median()
    iqr = q3 - q1
    minimum = data[col].min()
    maximum = data[col].max()

    print(f"\n{col}:")
    print(f"Mean: {mean:.2f}")
    print(f"SD: {std:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Range: {minimum:.2f} - {maximum:.2f}")

# %% Categorical Distribution
print("\nCategorical Distribution:")

categorical_cols = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

for col in categorical_cols:
    print(f"\n{col}:")

    counts = data[col].value_counts(dropna=False)
    percentages = (
        data[col].value_counts(normalize=True, dropna=False) * 100
    )

    category_summary = pd.DataFrame({
        "Count": counts,
        "Percentage (%)": percentages.round(2)
    })

    print(category_summary)


# %% Show Histogram for All Columns

# Create folder to save figures
save_dir = Path("figures/histograms")
save_dir.mkdir(parents=True, exist_ok=True)

columns = data.columns

for col in columns:
    print("col:", col)

    plt.figure(figsize=(8, 5))

    if pd.api.types.is_numeric_dtype(data[col]):
        data[col].hist(bins=30)
    else:
        data[col].value_counts(dropna=False).plot(kind="bar")
        plt.xticks(rotation=45, ha="right")

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()

    # Save each figure separately
    file_path = save_dir / f"{col}_distribution.png"

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {file_path}")

    plt.show()
    plt.close()

# %% Numerical Features by Stroke Status
print("\nNumerical Features by Stroke Status:")

# Create folder to save box plots
save_dir = Path("figures/stroke_comparisons")
save_dir.mkdir(parents=True, exist_ok=True)

for col in numerical_cols:
    data.boxplot(
        column=col,
        by="stroke",
        figsize=(7, 5)
    )

    plt.title(f"{col} by Stroke Status")
    plt.suptitle("")
    plt.xlabel("Stroke Status (0 = No, 1 = Yes)")
    plt.ylabel(col)
    plt.tight_layout()

    # Save each box plot separately
    file_path = save_dir / f"{col}_by_stroke.png"

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {file_path}")

    plt.show()
    plt.close()

# %% Show Preprocessed DataFrame
X_train, X_test, y_train, y_test = data_loader.get_data_split()

print("\nPreprocessed Training Shape:")
print(X_train.shape)

print("\nFirst 5 Rows After Preprocessing:")
print(X_train.head())

print("\nPreprocessed Testing Shape:")
print(X_test.shape)

