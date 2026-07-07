from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

from .paths import TABLES_DIR


PRIMARY_METRIC = "wmae"

HOLIDAY_WEIGHT = 5.0
NON_HOLIDAY_WEIGHT = 1.0

DEFAULT_RANDOM_SEED = 42

CV_FOLDS = 3
CV_VALIDATION_WEEKS = 13
FINAL_HOLDOUT_WEEKS = 39

VALIDATION_ID = "expanding_3x13_holdout_39_v1"
DATA_VERSION = "processed_v1"

ALLOWED_STAGES = {
    "baseline",
    "tuning",
    "final",
}

ALLOWED_EVALUATION_SCOPES = {
    "full_dataset",
    "subset",
    "representative_series",
}


RESULT_COLUMNS = [
    "architecture",
    "run_name",
    "stage",
    "tracker",
    "feature_set",
    "preprocessing",
    "validation_id",
    "data_version",
    "evaluation_scope",
    "forecast_strategy",
    "series_count",
    "random_seed",
    "cv_wmae_mean",
    "cv_wmae_std",
    "holdout_wmae",
    "holdout_mae",
    "holdout_rmse",
    "fit_seconds",
    "predict_seconds",
    "notes",
]


def slugify(value: str) -> str:
    value = value.strip().lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    return value.strip("_")


def make_run_name(
    architecture: str,
    stage: str,
    feature_set: str,
    trial_name: str,
    seed: int = DEFAULT_RANDOM_SEED,
) -> str:
    if stage not in ALLOWED_STAGES:
        raise ValueError(
            f"Unknown experiment stage: {stage}"
        )

    parts = [
        architecture,
        stage,
        feature_set,
        trial_name,
        f"s{seed}",
    ]

    return "__".join(
        slugify(part)
        for part in parts
    )


def build_common_parameters(
    *,
    architecture: str,
    stage: str,
    feature_set: str,
    preprocessing: str,
    evaluation_scope: str,
    forecast_strategy: str,
    series_count: int,
    random_seed: int = DEFAULT_RANDOM_SEED,
    extra_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if stage not in ALLOWED_STAGES:
        raise ValueError(
            f"Unknown experiment stage: {stage}"
        )

    if (
        evaluation_scope
        not in ALLOWED_EVALUATION_SCOPES
    ):
        raise ValueError(
            "Unknown evaluation scope: "
            f"{evaluation_scope}"
        )

    parameters = {
        "architecture": architecture,
        "stage": stage,
        "feature_set": feature_set,
        "preprocessing": preprocessing,
        "target": "Weekly_Sales",
        "forecast_key": "Store+Dept+Date",
        "primary_metric": PRIMARY_METRIC,
        "holiday_weight": HOLIDAY_WEIGHT,
        "non_holiday_weight": (
            NON_HOLIDAY_WEIGHT
        ),
        "validation_id": VALIDATION_ID,
        "cv_folds": CV_FOLDS,
        "cv_validation_weeks": (
            CV_VALIDATION_WEEKS
        ),
        "final_holdout_weeks": (
            FINAL_HOLDOUT_WEEKS
        ),
        "data_version": DATA_VERSION,
        "evaluation_scope": evaluation_scope,
        "forecast_strategy": forecast_strategy,
        "series_count": series_count,
        "random_seed": random_seed,
    }

    if extra_parameters:
        parameters.update(
            dict(extra_parameters)
        )

    return parameters


def build_result_row(
    *,
    architecture: str,
    run_name: str,
    stage: str,
    tracker: str,
    feature_set: str,
    preprocessing: str,
    evaluation_scope: str,
    forecast_strategy: str,
    series_count: int,
    metrics: Mapping[str, float],
    fit_seconds: float,
    predict_seconds: float,
    random_seed: int = DEFAULT_RANDOM_SEED,
    notes: str = "",
) -> dict[str, Any]:
    required_metrics = {
        "cv_wmae_mean",
        "cv_wmae_std",
        "holdout_wmae",
        "holdout_mae",
        "holdout_rmse",
    }

    missing_metrics = (
        required_metrics - metrics.keys()
    )

    if missing_metrics:
        raise ValueError(
            "Missing result metrics: "
            f"{sorted(missing_metrics)}"
        )

    return {
        "architecture": architecture,
        "run_name": run_name,
        "stage": stage,
        "tracker": tracker,
        "feature_set": feature_set,
        "preprocessing": preprocessing,
        "validation_id": VALIDATION_ID,
        "data_version": DATA_VERSION,
        "evaluation_scope": evaluation_scope,
        "forecast_strategy": forecast_strategy,
        "series_count": series_count,
        "random_seed": random_seed,
        "cv_wmae_mean": metrics[
            "cv_wmae_mean"
        ],
        "cv_wmae_std": metrics[
            "cv_wmae_std"
        ],
        "holdout_wmae": metrics[
            "holdout_wmae"
        ],
        "holdout_mae": metrics[
            "holdout_mae"
        ],
        "holdout_rmse": metrics[
            "holdout_rmse"
        ],
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "notes": notes,
    }


def save_architecture_result(
    result: Mapping[str, Any],
) -> None:
    architecture = slugify(
        str(result["architecture"])
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_frame = pd.DataFrame(
        [result],
        columns=RESULT_COLUMNS,
    )

    result_frame.to_csv(
        TABLES_DIR
        / f"{architecture}_final.csv",
        index=False,
    )