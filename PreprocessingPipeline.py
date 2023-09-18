import pandas as pd
import Imputer
from sklearn.preprocessing import Normalizer


class PreprocessingPipeline:

    def __init__(self, data: pd.DataFrame, copy: bool = True):
        self.data = data.copy() if copy else data
        self.imputer = Imputer.Imputer(self.data)

    def correlation_matrix(self, full_matrix: bool = False) -> pd.DataFrame:
        try:
            if not full_matrix:
                return self.data.corr()['target'].sort_values(ascending=False)
            else:
                return self.data.corr()
        except ValueError:
            print("The dataframe contains categorical values.")
            return None

    def transpose_correlation_matrix(self, data: pd.DataFrame,
                                     file_name: str) -> pd.DataFrame:

        data.to_csv(f'dcmi/correlation_matrixes/{file_name}.csv', index=True)

        corr = pd.DataFrame(pd.read_csv(
            f'dcmi/correlation_matrixes/{file_name}.csv'))

        corr = corr.transpose()

        corr.to_csv(
            f'dcmi/correlation_matrixes/{file_name}.csv',
            index=False, header=False)

        corr = pd.DataFrame(pd.read_csv(
            f'dcmi/correlation_matrixes/{file_name}.csv'))

        return corr

    @staticmethod
    def split_labels(data, label: str = 'target') -> pd.DataFrame:
        labels = data.pop(label)
        return labels

    def remove_null_values(self, save_csv: bool = False) -> pd.DataFrame:
        data = self.imputer.remove_null_values(save_csv=save_csv)
        return data

    def fill_null_values_with_0(self, save_csv: bool = False) -> pd.DataFrame:
        data = self.imputer.fill_null_values_with_0(save_csv=save_csv)
        return data

    def fill_null_values_with_mean(self,
                                   save_csv: bool = False) -> pd.DataFrame:

        data = self.imputer.fill_null_values_with_mean(save_csv=save_csv)
        return data

    def fill_null_values_with_median(self,
                                     save_csv: bool = False) -> pd.DataFrame:

        data = self.imputer.fill_null_values_with_median(save_csv=save_csv)
        return data

    @staticmethod
    def normalizer(data) -> pd.DataFrame:
        transformer = Normalizer().fit(data)
        data = pd.DataFrame(transformer.transform(data))
        return data
