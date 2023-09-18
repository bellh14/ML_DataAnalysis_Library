import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from Dataset import Dataset


class Models:

    def __init__(self, data: Dataset):
        self.data = data

    def linear_regression(self) -> LinearRegression.Model:
        model = LinearRegression()
        model.fit(self.data.train_data, self.data.train_labels)
        return model

    def random_forest(self, n_estimaters: int = 1000, max_depth: int = 10,
                      n_jobs: int = -1, random_state: int = None
                      ) -> RandomForestRegressor.Model:
        model = RandomForestRegressor(n_estimaters=n_estimaters,
                                      max_depth=max_depth, n_jobs=n_jobs,
                                      random_state=random_state)
        model.fit(self.data.train_data, self.data.train_labels)
        return model

    def build_dnn_model(self, input_shape: int, output_shape: int,
                        optimizer: str = 'adam', loss: str = 'mse',
                        metrics: list = ['mae', 'mse']) -> tf.keras.Model:
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[input_shape]),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)
        return model

    def dnn_model(self, epochs: int = 1000):
        dnn_model = self.build_dnn_model(input_shape=len(
                                         self.data.train_data.keys()),
                                         output_shape=1)
        history = dnn_model.fit(self.data.train_data, self.data.train_labels,
                                epochs=epochs, validation_split=0.2,
                                verbose=0)
        return history, dnn_model
