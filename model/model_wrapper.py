import pandas as pd
import os

from model.data_processor import DataProcessor
from model.model_loader import ModelLoader
from model.predictor import Predictor

class ModelWrapper:
    def __init__(self,
                 model_path="best_model.pkl",
                 threshold_path="best_threshold.json",
                 train_path="data/heart_train.csv"):
        # загрузка модели
        self.loader = ModelLoader(model_path, threshold_path)
        self.model = self.loader.model
        self.threshold = self.loader.threshold

        # инициализация предиктора
        self.predictor = Predictor(self.model, self.threshold)

        # препроцессор + fit
        self.processor = DataProcessor()

        if train_path and os.path.exists(train_path):
            df_train = pd.read_csv(train_path)
            self.processor.fit(df_train)
        else:
            self.processor.fit(pd.DataFrame())

    def predict_from_df(self, df: pd.DataFrame):
        if "id" not in df.columns:
            raise ValueError("Input DataFrame must contain 'id' column")

        df = self.processor.transform(df)
        ids = df["id"].astype(int).tolist()
        X = df.drop(columns=["id"])

        preds = self.predictor.predict(X)

        return ids, preds