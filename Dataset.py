import pandas as pd
import numpy as np


class Dataset:
    def __init__(
        self,
        dataset_name: str,
        train_file_name: str,
        folder_name: str,
        test_file_name: str,
        label_name: str,
        validation_split: float = 0.2,
    ):
        self.dataset_name = dataset_name
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.folder_name = folder_name

        self.train_data = self.load_train_data()
        self.test_data = self.load_test_data()

        self.label_name = label_name
        self.train_labels = None
        self.test_labels = None
        self.validation_split = validation_split

        self.categorial_columns = self.get_categorical_columns()
        self.numerical_columns = self.get_numerical_columns()

    def load_train_data(self) -> pd.DataFrame:
        return pd.DataFrame(pd.read_csv(f"{self.folder_name}/{self.train_file_name}"))

    def load_test_data(self) -> pd.DataFrame:
        return pd.DataFrame(pd.read_csv(f"{self.folder_name}/{self.test_file_name}"))

    def save_csv_data(self, file_name: str) -> None:
        self.train_data.to_csv(f"{self.folder_name}/{file_name}_train.csv", index=False)
        self.test_data.to_csv(f"{self.folder_name}/{file_name}_test.csv", index=False)

    def update_train_data(self, data: pd.DataFrame) -> None:
        self.train_data = data

    def update_test_data(self, data: pd.DataFrame) -> None:
        self.test_data = data

    def transpose(self) -> pd.DataFrame:
        self.data = self.data.T

    def describe(self, save_csv: bool = False) -> None:
        print(self.data.describe())
        if save_csv:
            self.save_csv(f"{self.file_name}_describe", self.data.describe())

    def get_null_values(self) -> pd.DataFrame:
        return self.data.isnull().sum()

    def get_correlation_matrix(self, full_mat: bool = False) -> pd.DataFrame:
        try:
            if not full_mat:
                return self.data.corr()["target"].sort_values(ascending=False)
            else:
                return self.data.corr()
        except ValueError:
            print("The dataframe contains categorical values.")
            return None

    def split_validation_data(self, validation_split: float) -> pd.Dataframe:
        train_data = self.data.sample(frac=validation_split)
        validation_data = self.data.drop(train_data.index)
        return train_data, validation_data

    def get_categorical_columns(self) -> pd.DataFrame:
        return self.train_data.select_dtypes(include=[object]).columns

    def get_numerical_columns(self) -> pd.DataFrame:
        return self.train_data.select_dtypes(include=[np.number]).columns
