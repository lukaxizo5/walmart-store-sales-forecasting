from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Iterator

import dagshub
import mlflow
import wandb
from dotenv import load_dotenv

from .paths import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env")


def get_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")

    return value


def setup_mlflow(experiment_name: str) -> None:
    dagshub.init(
        repo_owner=get_env("DAGSHUB_REPO_OWNER"),
        repo_name=get_env("DAGSHUB_REPO_NAME"),
        mlflow=True,
    )

    mlflow.set_experiment(experiment_name)


def prepare_parameters(
    parameters: dict[str, Any],
) -> dict[str, Any]:
    prepared = {}

    for key, value in parameters.items():
        if isinstance(value, (list, tuple, dict)):
            prepared[key] = json.dumps(value, default=str)
        else:
            prepared[key] = value

    return prepared


@contextmanager
def tracked_run(
    experiment_name: str,
    run_name: str,
    parameters: dict[str, Any] | None = None,
    wandb_group: str | None = None,
    job_type: str = "training",
    use_wandb: bool = True,
) -> Iterator[Any | None]:
    setup_mlflow(experiment_name)

    parameters = parameters or {}
    mlflow_parameters = prepare_parameters(parameters)

    with mlflow.start_run(run_name=run_name):
        if mlflow_parameters:
            mlflow.log_params(mlflow_parameters)

        if not use_wandb:
            yield None
            return

        with wandb.init(
            entity=get_env("WANDB_ENTITY"),
            project=get_env("WANDB_PROJECT"),
            name=run_name,
            group=wandb_group,
            job_type=job_type,
            config=parameters,
        ) as wandb_run:
            yield wandb_run


def log_metrics(
    metrics: dict[str, float],
    wandb_run: Any | None = None,
    step: int | None = None,
) -> None:
    metrics = {
        name: float(value)
        for name, value in metrics.items()
    }

    if step is None:
        mlflow.log_metrics(metrics)
    else:
        mlflow.log_metrics(metrics, step=step)

    if wandb_run is not None:
        if step is None:
            wandb_run.log(metrics)
        else:
            wandb_run.log(metrics, step=step)


def log_artifact(
    path: str,
    artifact_path: str | None = None,
) -> None:
    mlflow.log_artifact(
        path,
        artifact_path=artifact_path,
    )