import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

pd.set_option("display.max_columns", None)


# %% Reusable Preprocessing
def build_preprocessor(X):
    categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int), categorical_cols)
    ], verbose_feature_names_out=False)


# %% Reusable Model Pipeline
def build_pipeline(X, classifier, random_state=2021):
    return Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("oversampler", RandomOverSampler(sampling_strategy="minority", random_state=random_state)),
        ("classifier", classifier)
    ])


# %% DataLoader
class DataLoader:
    def __init__(self):
        self.data = None

    def load_dataset(self, path="data/healthcare-dataset-stroke-data.csv"):
        self.data = pd.read_csv(path)
        return self.data

    def remove_duplicates(self):
        # Remove duplicate patient IDs and duplicate records excluding ID.
        # Keep the first occurrence.
        self.data = self.data.drop_duplicates(subset=["id"]).copy()
        self.data = self.data.drop_duplicates(subset=self.data.columns.drop("id")).reset_index(drop=True)
        return self.data

    def get_data_split(self, test_size=0.20, random_state=2021):
        X = self.data.drop(columns=["id", "stroke"])
        y = self.data["stroke"]

        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    def get_train_val_test_split(self, test_size=0.20, validation_size=0.20, random_state=2021):
        X_train_val, X_test, y_train_val, y_test = self.get_data_split(test_size, random_state)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=validation_size,
            random_state=random_state,
            stratify=y_train_val
        )

        return X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val

    def preprocess_data(self, X_train, X_test):
        preprocessor = build_preprocessor(X_train)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        feature_names = preprocessor.get_feature_names_out()

        X_train = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
        X_test = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

        return X_train, X_test

    def oversample(self, X_train, y_train, random_state=2021):
        oversampler = RandomOverSampler(sampling_strategy="minority", random_state=random_state)
        return oversampler.fit_resample(X_train, y_train)