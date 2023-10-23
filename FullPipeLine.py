from Dataset import Dataset
from PreprocessingPipeline import PreprocessingPipeline
from Models import Models
from Visualizer import Visualizer
from HyperParameterTuner import HyperParameterTuner


class FullPipeLine:
    def __init__(
        self,
        data: Dataset,
        imputation_strategy: str = "0",
        normalization_method: str = "l2",
        model_type: str = "linear_regression",
        null_threshold: float = None,
        visualize: bool = True,
    ):
        self.data = data
        self.imputation_strategy = imputation_strategy
        self.model_type = model_type
        self.null_threshold = null_threshold
        self.visualize = visualize

    def run_pipeline(self):
        preprocesing_pipeline = PreprocessingPipeline(self.data)
        preprocesing_pipeline.fill_null_values_with_0()
        preprocesing_pipeline.normalizer()
        preprocesing_pipeline.split_labels()

        model = Models(self.data)
        if self.model_type == "linear_regression":
            trained_model = model.linear_regression()
        elif self.model_type == "random_forest":
            trained_model = model.random_forest()
        elif self.model_type == "dnn":
            trained_model = model.dnn_model()
