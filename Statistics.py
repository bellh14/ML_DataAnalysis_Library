import pandas as pd
import numpy as np


class Statistics:
    def __init__(
        self, data: pd.DataFrame, file_name: str = None, folder_name: str = "dcmi"
    ):
        self.data = data
        self.file_name = file_name
        self.folder_name = folder_name

    def save_csv(self, file_name: str, other_data: pd.DataFrame = None):
        if other_data is None:
            self.data.to_csv(f"dcmi/{file_name}.csv", index=False)
        else:
            other_data.to_csv(f"dcmi/{file_name}.csv", index=False)

    def describe(self, save_csv: bool = False) -> None:
        print(self.data.describe())
        if save_csv:
            self.save_csv(f"{self.file_name}_describe", self.data.describe())

    def get_null_values(self) -> pd.DataFrame:
        return self.data.isnull().sum()
