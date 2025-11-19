# ml/model_loader.py
import joblib
import json

class ModelLoader:
    def __init__(self, model_path, threshold_path):
        self.model = joblib.load(model_path)
        with open(threshold_path, "r") as f:
            self.threshold = json.load(f)["threshold"]

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]