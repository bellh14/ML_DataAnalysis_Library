import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def scatter_plot(self, x: str = None, y: str = None, color: str = None, size: str = None):
        fig = px.scatter(self.data, x=x, y=y, color=color, size=size)
        fig.show()

    def plt_bar_plot(self, x: str = None, y: str = None):
        plt.bar(x, y)
        plt.show()