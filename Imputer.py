import pandas as pd
from Dataset import Dataset


class Imputer:
    def __init__(self, data: Dataset):
        self.data = data
        self.base_file_name = self.data.dataset_name
        self.folder = self.data.folder_name

    def run_save_all(self):
        self.remove_null_values(save_csv=True)
        self.fill_null_values_with_0(save_csv=True)
        self.fill_null_values_with_mean(save_csv=True)
        self.fill_null_values_with_median(save_csv=True)

    def remove_null_values(self, save_csv: bool = False) -> pd.DataFrame:
        self.data.dropna(inplace=True)

        if save_csv:
            self.data.save_csv_data(f"{self.folder}/{self.base_file_name}_remove_null")

        return self.data

    def fill_null_values_with_0(self, save_csv: bool = False) -> pd.DataFrame:
        self.data.fillna(0, inplace=True)

        if save_csv:
            self.data.save_csv_data(f"{self.folder}/{self.base_file_name}_null_0_fill")

        return self.data

    def fill_null_values_with_mean(self, save_csv: bool = False) -> pd.DataFrame:
        self.data.fillna(self.data.mean(), inplace=True)

        if save_csv:
            self.data.save_csv_data(
                f"{self.folder}/{self.base_file_name}_null_mean_fill"
            )

        return self.data

    def fill_null_values_with_median(self, save_csv: bool = False) -> pd.DataFrame:
        self.data.fillna(self.data.median(), inplace=True)

        if save_csv:
            self.data.save_csv_data(
                f"{self.folder}/{self.base_file_name}_null_median_fill"
            )

        return self.data

    def fill_null_values_with_custom(
        self, value: str, save_csv: bool = False
    ) -> pd.DataFrame:
        self.data.fillna(value, inplace=True)

        if save_csv:
            self.data.save_csv_data(
                f"{self.folder}/{self.base_file_name}_null_{value}_fill"
            )

        return self.data

    def fill_null_values_with_mode(self, save_csv: bool = False) -> pd.DataFrame:
        self.data.fillna(self.data.mode(), inplace=True)

        if save_csv:
            self.data.save_csv_data(
                f"{self.folder}/{self.base_file_name}_null_mode_fill"
            )

        return self.data
