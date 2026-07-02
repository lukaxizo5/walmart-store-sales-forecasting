from __future__ import annotations

import dagshub
import mlflow
import wandb


DAGSHUB_REPO_OWNER = "lkhiz23"
DAGSHUB_REPO_NAME = "walmart-store-sales-forecasting"

WANDB_ENTITY = "lkhiz23-free-university-of-tbilisi-"
WANDB_PROJECT = "walmart-store-sales-forecasting"

MLFLOW_EXPERIMENT = "walmart-store-sales-forecasting"


def main() -> None:
    # Configure MLflow to use the DagsHub repository's tracking server.
    dagshub.init(
        repo_owner=DAGSHUB_REPO_OWNER,
        repo_name=DAGSHUB_REPO_NAME,
        mlflow=True,
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    parameters = {
        "model": "dummy-baseline",
        "seed": 42,
        "purpose": "connection-test",
    }

    metrics = {
        "validation_wmae": 12345.67,
        "validation_mae": 11000.00,
    }

    run_name = "tracking-connection-test"

    # Log the test to DagsHub/MLflow.
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "model_family": "test",
                "author": "lkhiz23",
            }
        )

        # Log the same test to W&B.
        with wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=run_name,
            group="setup",
            job_type="connection-test",
            config=parameters,
        ) as wandb_run:
            wandb_run.log(metrics)

    print("Successfully logged a test run to DagsHub/MLflow and W&B.")


if __name__ == "__main__":
    main()