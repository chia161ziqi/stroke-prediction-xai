import pandas as pd 
# Makes sure we see all columns
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler

class DataLoader():
    def __init__(self):
        self.data = None

    def load_dataset(self, path="data/healthcare-dataset-stroke-data.csv"):
        self.data = pd.read_csv(path)

    def preprocess_data(self, X_train, X_test):
        categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

        X_train = X_train.copy()
        X_test = X_test.copy()

        # BMI imputation: fit on training data only
        imputer = SimpleImputer(strategy="median")
        X_train["bmi"] = imputer.fit_transform(X_train[["bmi"]]).ravel()
        X_test["bmi"] = imputer.transform(X_test[["bmi"]]).ravel()

        # One-hot encoding: fit on training data only
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int)
        train_encoded = encoder.fit_transform(X_train[categorical_cols])
        test_encoded = encoder.transform(X_test[categorical_cols])
        encoded_cols = encoder.get_feature_names_out(categorical_cols)

        train_encoded = pd.DataFrame(train_encoded, columns=encoded_cols, index=X_train.index)
        test_encoded = pd.DataFrame(test_encoded, columns=encoded_cols, index=X_test.index)

        # Combine encoded and numerical features
        X_train = pd.concat([train_encoded, X_train.drop(columns=categorical_cols)], axis=1)
        X_test = pd.concat([test_encoded, X_test.drop(columns=categorical_cols)], axis=1)

        return X_train, X_test

    def get_data_split(self):
        X = self.data.drop(columns=["id", "stroke"])
        y = self.data["stroke"]
        return train_test_split(X, y, test_size=0.20, random_state=2021, stratify=y)
    
    def oversample(self, X_train, y_train):
        oversample = RandomOverSampler(sampling_strategy='minority')
        # Convert to numpy and oversample
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        x_np, y_np = oversample.fit_resample(x_np, y_np)
        # Convert back to pandas
        x_over = pd.DataFrame(x_np, columns=X_train.columns)
        y_over = pd.Series(y_np, name=y_train.name)
        return x_over, y_over

