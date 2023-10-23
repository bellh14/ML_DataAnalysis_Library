import pandas as pd
import Imputer
from sklearn.preprocessing import Normalizer, MinMaxScaler
from Dataset import Dataset
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


class PreprocessingPipeline:
    def __init__(self, data: Dataset):
        self.data = data.copy()
        self.imputer = Imputer.Imputer(self.data)

    def remove_null_values(self, save_csv: bool = False) -> pd.DataFrame:
        data = self.imputer.remove_null_values(save_csv=save_csv)
        return data

    def fill_null_values_with_0(self, save_csv: bool = False) -> pd.DataFrame:
        data = self.imputer.fill_null_values_with_0(save_csv=save_csv)
        return data

    def fill_null_values_with_mean(self, save_csv: bool = False) -> pd.DataFrame:
        data = self.imputer.fill_null_values_with_mean(save_csv=save_csv)
        return data

    def fill_null_values_with_median(self, save_csv: bool = False) -> pd.DataFrame:
        data = self.imputer.fill_null_values_with_median(save_csv=save_csv)
        return data

    def fill_null_values_with_custom(
        self,
        value: str,
    ) -> pd.DataFrame:
        data = self.imputer.fill_null_values_with_custom(value)
        return data

    def normalizer(self, normalization_method: str = "l2") -> None:
        train_transformer = Normalizer(normalization_method).fit(self.data.train_data)
        train_transformer.transform(self.data.train_data)
        test_transformer = Normalizer(normalization_method).fit(self.data.test_data)
        test_transformer.transform(self.data.test_data)

    def min_max_scaler(self) -> None:
        scaler = MinMaxScaler()
        scaler.fit(self.data.train_data)
        scaler.transform(self.data.train_data)

    def drop_feature(self, feature: str) -> None:
        self.data.train_data.drop(feature, axis=1, inplace=True)
        self.data.test_data.drop(feature, axis=1, inplace=True)

    def drop_features(self, features: list) -> None:
        for feature in features:
            self.drop_feature(feature)

    def drop_by_null_threshold(self, threshold: float) -> None:
        self.data.train_data.dropna(thresh=threshold, axis=1, inplace=True)
        self.data.test_data.dropna(thresh=threshold, axis=1, inplace=True)

    def ordinal_encode(self) -> None:
        ordinal_encoder = OrdinalEncoder()
        ordinal_encoder.fit(self.data.train_data)
        ordinal_encoder.transform(self.data.train_data)
        ordinal_encoder.transform(self.data.test_data)

    def one_hot_encode(self) -> None:
        one_hot_encoder = OneHotEncoder()
        one_hot_encoder.fit(self.data.train_data)
        one_hot_encoder.transform(self.data.train_data)
        one_hot_encoder.transform(self.data.test_data)

    def split_labels(self) -> None:
        self.data.train_labels = self.data.train_data.pop(self.data.label_name)
        self.data.test_labels = self.data.test_data.pop(self.data.label_name)
