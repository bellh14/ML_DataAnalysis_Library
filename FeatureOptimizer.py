import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import Visualizer


class FeatureOptimizer:

    def get_null_values(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.isnull().sum()

    def get_null_values_percentage(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.isnull().sum() / len(data) * 100

    def convert(self, data: pd.DataFrame):
        for col in data.columns:
            data[col] = np.where(data[col] < 0, 0, data[col])
        return data

    def training_model(self, train: pd.DataFrame):
        # train is your submission!
        train_labels = train.pop('target')
        rf = RandomForestRegressor(
            n_estimators=1000,
            max_depth=7,
            n_jobs=-1,
            random_state=42)
        rf.fit(train, train_labels)

        # predictions = rf.predict(self.test)
        # print("Accuracy:", rf.score(self.test, self.test_labels))
        # feature_importances = pd.DataFrame(pd.concat([pd.DataFrame(np.transpose(rf.feature_names_in_)), pd.DataFrame(
        #     np.transpose(rf.feature_importances_))], axis=1), columns=['feature', 'importance'])
        feature_importances = pd.DataFrame(pd.concat([pd.DataFrame(rf.feature_names_in_), pd.DataFrame(
            rf.feature_importances_)], axis=1))
        feature_importances.columns = ['feature', 'importance']
        # feature_importances = np.transpose(feature_importances)
        print(feature_importances)
        Visualizer.Visualizer(feature_importances).plt_bar_plot(
            x=feature_importances['feature'], y=feature_importances['importance'])


if __name__ == "__main__":
    data = pd.read_csv("dcmi/sample_submission.csv")
    fo = FeatureOptimizer()
    fo.training_model(data)
