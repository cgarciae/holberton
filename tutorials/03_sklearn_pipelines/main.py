#
# Typer: https://github.com/tiangolo/typer
# MLFlow: https://mlflow.org/
# Sckit Learn: https://scikit-learn.org/stable/data_transforms.html
#

import os
import pickle
import typing as tp
from dataclasses import dataclass
from random import randint, random
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import typer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.minmax_scaler = MinMaxScaler()
        self.std_scaler = StandardScaler()

    def fit(self, x) -> "Preprocessor":

        self.minmax_scaler.fit(x[:, 0])
        self.std_scaler.fit(x[:, 1])

        return self

    def transform(self, x) -> np.ndarray:

        x0 = self.minmax_scaler.transform(x[:, 0])
        x1 = self.std_scaler.transform(x[:, 1])

        x = np.stack([x0, x1], axis=1)

        return x


class Model(tp.NamedTuple):
    batch_size: int
    learning_rate: float
    transformer: tp.Any

    def fit(self, x):
        # transform data
        self.transformer.fit(x)
        x = self.transformer.transform(x)

        # training
        # ....

        return dict(accuracy=random())

    def eval(self, x):
        # transform data
        x = self.transformer.transform(x)

        # eval
        # .....
        return dict(accuracy=random())

    def predict(self, x):
        # transform data
        x = self.transformer.transform(x)

        # predict
        y = x

        return y

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)


def load_data() -> np.ndarray:
    return np.random.uniform(-10, 10, size=(100, 2))


def split_data(data: np.ndarray, split: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    n = int(len(data) * split)
    return data[:n], data[n:]


def main(
    batch_size: int = typer.Option(64, help="Size of the batch for training"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate."),
    split: float = typer.Option(0.8, help="Learning rate."),
):
    data = load_data()

    x_train, x_test = split_data(data, split)

    model = Model(
        batch_size=batch_size,
        learning_rate=learning_rate,
        transformer=ColumnTransformer(
            [
                ("first", MinMaxScaler(), 0),
                ("second", StandardScaler(), 1),
            ],
        ),
        # transformer=Preprocessor(),
    )

    metrics = model.fit(x_train)
    metrics = model.eval(x_test)

    # Log a parameter (key-value pair)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)

    # Log a metric; metrics can be updated throughout the run
    mlflow.log_metric("accuracy", metrics["accuracy"])

    with TemporaryDirectory() as output_path:

        model.save(f"{output_path}/model.pkl")

        mlflow.log_artifact("tutorials/02_mlflow/main.py")
        mlflow.log_artifacts(output_path)

    print(model)


if __name__ == "__main__":
    typer.run(main)
