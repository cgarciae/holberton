#
# Typer: https://github.com/tiangolo/typer
# MLFlow: https://mlflow.org/
#

import typer
import typing as tp

import os
from random import random, randint
import mlflow
import pickle
from tempfile import TemporaryDirectory


class Model(tp.NamedTuple):
    batch_size: int
    learning_rate: float

    def fit(self):
        return dict(accuracy=random())

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)


def main(
    batch_size: int = typer.Option(64, help="Size of the batch for training"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate."),
):
    model = Model(batch_size=batch_size, learning_rate=learning_rate)

    metrics = model.fit()

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