#
# Typer: https://github.com/tiangolo/typer
#

import typer
import typing as tp


class Model(tp.NamedTuple):
    batch_size: int
    learning_rate: float


def main(
    batch_size: int = typer.Option(64, help="Size of the batch for training"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate."),
):
    model = Model(batch_size=batch_size, learning_rate=learning_rate)

    print(model)


if __name__ == "__main__":
    typer.run(main)