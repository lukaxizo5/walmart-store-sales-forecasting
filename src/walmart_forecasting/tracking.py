from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import dagshub
import mlflow
import wandb
from dotenv import load_dotenv

from .paths import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env")


def get_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Missing environment variable: {name}"
        )

    return value


def prepare_parameters(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    prepared = {}

    for name, value in parameters.items():
        if isinstance(value, (list, tuple, dict)):
            prepared[name] = json.dumps(
                value,
                default=str,
                sort_keys=True,
            )
        else:
            prepared[name] = value

    return prepared


def setup_mlflow(experiment_name: str) -> None:
    dagshub.init(
        repo_owner=get_env("DAGSHUB_REPO_OWNER"),
        repo_name=get_env("DAGSHUB_REPO_NAME"),
        mlflow=True,
    )

    mlflow.set_experiment(experiment_name)


@contextmanager
def mlflow_run(
    experiment_name: str,
    run_name: str,
    parameters: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    setup_mlflow(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        if parameters:
            mlflow.log_params(
                prepare_parameters(parameters)
            )

        if tags:
            mlflow.set_tags(dict(tags))

        yield run


@contextmanager
def wandb_run(
    run_name: str,
    group: str,
    config: Mapping[str, Any] | None = None,
    job_type: str = "training",
    tags: list[str] | None = None,
) -> Iterator[Any]:
    with wandb.init(
        entity=get_env("WANDB_ENTITY"),
        project=get_env("WANDB_PROJECT"),
        name=run_name,
        group=group,
        job_type=job_type,
        config=dict(config or {}),
        tags=tags,
    ) as run:
        yield run