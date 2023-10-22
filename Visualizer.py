import plotly.express as px
import matplotlib.pyplot as plt
from Dataset import Dataset


class Visualizer:
    def __init__(self, data: Dataset):
        self.data = data

    def scatter_plot(
        self, x: str = None, y: str = None, color: str = None, size: str = None
    ):
        fig = px.scatter(self.data, x=x, y=y, color=color, size=size)
        fig.show()

    def plt_bar_plot(self, x: str = None, y: str = None):
        plt.bar(x, y)
        plt.show()

    def plot_loss_plt(self, history, ymin: int, ymax: int):
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.ylim([ymin, ymax])
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(f"{self.data.folder_name}{self.data.model_name}_loss.png")
