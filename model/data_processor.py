# model/data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import joblib
import os

class DataProcessor:
    def __init__(self):
        self.encoder = None
        self.cat_cols = None
        self.num_cols = None
        self.feature_order = None   # порядок признаков модели

    def fit(self, df: pd.DataFrame):
        df = df.copy()

        # --- 1. Сохранить порядок фичей модели ---
        # если файл есть — загружаем
        if os.path.exists("feature_order.pkl"):
            self.feature_order = joblib.load("feature_order.pkl")

        # если файл ещё не создан (первый запуск training)
        else:
            # в этом случае feature_order создаётся в train.py
            pass

        # --- 2. Определение колонок перед кодированием ---
        self.num_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()

        # --- 3. Обучение OrdinalEncoder ---
        if self.cat_cols:
            self.encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            self.encoder.fit(df[self.cat_cols])

        return self

    def transform(self, df: pd.DataFrame):
        df = df.copy()

        # --- Очистка ---
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        if "Unnamed: 0" in df.columns:
            df.drop(columns="Unnamed: 0", inplace=True)

        # --- Категориальные признаки ---
        if self.cat_cols:
            df[self.cat_cols] = self.encoder.transform(df[self.cat_cols]).astype(int)

        # --- Бинарные числовые ---
        for col in self.num_cols:
            if col in df.columns:
                if df[col].nunique() == 2 and df[col].dtype != int:
                    df[col] = df[col].astype(int)

        # =====================================================
        # ✔ Генерация тех же фичей, что были при обучении
        # =====================================================

        df["age_chol_mul"] = df["Age"] * df["Cholesterol"]
        df["sys_dias_mul"] = df["Systolic blood pressure"] * df["Diastolic blood pressure"]
        df["stress_sedent_mul"] = df["Stress Level"] * df["Sedentary Hours Per Day"]

        df["trig_to_activity"] = np.where(
            df["Physical Activity Days Per Week"] != 0,
            df["Triglycerides"] / df["Physical Activity Days Per Week"],
            0
        )

        df["sleep_stress_mul"] = df["Sleep Hours Per Day"] * df["Stress Level"]

        # --- Квадратичные признаки ---
        for col in df.columns:
            df[col + "_sq"] = df[col] ** 2

        # =====================================================
        # ✔ Привести порядок колонок к порядку модели
        # =====================================================

        if self.feature_order:
            missing = set(self.feature_order) - set(df.columns)
            if missing:
                raise ValueError(f"Не хватает признаков: {missing}")

            df = df[self.feature_order]

        return df

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)