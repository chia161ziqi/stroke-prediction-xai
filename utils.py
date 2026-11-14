import pandas as pd 
# Makes sure we see all columns
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler

# %% Reusable Preprocessing Pipeline
def build_preprocessor(X):
    categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int), categorical_cols)
    ], verbose_feature_names_out=False)

class DataLoader():
    def __init__(self):
        self.data = None

    def load_dataset(self, path="data/healthcare-dataset-stroke-data.csv"):
        self.data = pd.read_csv(path)

    def preprocess_data(self, X_train, X_test):
        preprocessor = build_preprocessor(X_train)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        feature_names = preprocessor.get_feature_names_out()

        X_train = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
        X_test = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

        return X_train, X_test

    def get_data_split(self):
        X = self.data.drop(columns=["id", "stroke"])
        y = self.data["stroke"]
        return train_test_split(X, y, test_size=0.20, random_state=2021, stratify=y)
    
    def oversample(self, X_train, y_train):
        oversample = RandomOverSampler(sampling_strategy='minority', random_state=2021)
        # Convert to numpy and oversample
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        x_np, y_np = oversample.fit_resample(x_np, y_np)
        # Convert back to pandas
        x_over = pd.DataFrame(x_np, columns=X_train.columns)
        y_over = pd.Series(y_np, name=y_train.name)
        return x_over, y_over

