from __future__ import annotations

import numpy as np
import mlflow

from walmart_forecasting.baselines import (
    SeasonalNaiveBaseline,
)
from walmart_forecasting.data import (
    load_processed_data,
)
from walmart_forecasting.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_mae,
)
from walmart_forecasting.tracking import (
    mlflow_run,
)
from walmart_forecasting.validation import (
    competition_like_holdout,
    expanding_window_splits,
)


def evaluate(
    model: SeasonalNaiveBaseline,
    training_data,
    validation_data,
) -> dict[str, float]:
    model.fit(training_data)

    predictions = model.predict(
        validation_data
    )

    return {
        "wmae": weighted_mae(
            y_true=validation_data[
                "Weekly_Sales"
            ],
            y_pred=predictions,
            is_holiday=validation_data[
                "IsHoliday"
            ],
        ),
        "mae": mean_absolute_error(
            y_true=validation_data[
                "Weekly_Sales"
            ],
            y_pred=predictions,
        ),
        "rmse": root_mean_squared_error(
            y_true=validation_data[
                "Weekly_Sales"
            ],
            y_pred=predictions,
        ),
        "seasonal_coverage": (
            model.last_seasonal_coverage
        ),
    }


def main() -> None:
    data = load_processed_data()

    final_split = competition_like_holdout(
        train_data=data.train,
        test_data=data.test,
    )

    cv_splits = expanding_window_splits(
        data=final_split.train,
        n_splits=3,
        validation_weeks=13,
    )

    fold_metrics = []

    for fold_number, fold in enumerate(
        cv_splits,
        start=1,
    ):
        model = SeasonalNaiveBaseline(
            seasonal_lag=52
        )

        metrics = evaluate(
            model=model,
            training_data=fold.train,
            validation_data=fold.validation,
        )

        fold_metrics.append(metrics)

        print(
            f"Fold {fold_number}: "
            f"WMAE={metrics['wmae']:.2f}, "
            f"coverage="
            f"{metrics['seasonal_coverage']:.2%}"
        )

    final_model = SeasonalNaiveBaseline(
        seasonal_lag=52
    )

    holdout_metrics = evaluate(
        model=final_model,
        training_data=final_split.train,
        validation_data=final_split.validation,
    )

    logged_metrics = {
        "cv_wmae_mean": float(
            np.mean(
                [
                    metrics["wmae"]
                    for metrics in fold_metrics
                ]
            )
        ),
        "cv_wmae_std": float(
            np.std(
                [
                    metrics["wmae"]
                    for metrics in fold_metrics
                ]
            )
        ),
        "holdout_wmae": (
            holdout_metrics["wmae"]
        ),
        "holdout_mae": (
            holdout_metrics["mae"]
        ),
        "holdout_rmse": (
            holdout_metrics["rmse"]
        ),
        "holdout_seasonal_coverage": (
            holdout_metrics[
                "seasonal_coverage"
            ]
        ),
    }

    for fold_number, metrics in enumerate(
        fold_metrics,
        start=1,
    ):
        logged_metrics[
            f"fold_{fold_number}_wmae"
        ] = metrics["wmae"]

        logged_metrics[
            f"fold_{fold_number}_coverage"
        ] = metrics["seasonal_coverage"]

    parameters = {
        "model": "SeasonalNaive",
        "seasonal_lag": 52,
        "cv_folds": 3,
        "cv_validation_weeks": 13,
        "final_holdout_weeks": (
            final_split.validation_weeks
        ),
        "fallback": (
            "pair_mean_then_department_"
            "mean_then_store_mean_"
            "then_global_mean"
        ),
    }

    with mlflow_run(
        experiment_name="Baseline_Training",
        run_name="SeasonalNaive_52",
        parameters=parameters,
        tags={
            "model_family": "baseline",
            "tracker": "mlflow",
        },
    ):
        mlflow.log_metrics(logged_metrics)

    print()
    print("Final holdout:")
    print(
        f"WMAE: "
        f"{holdout_metrics['wmae']:.2f}"
    )
    print(
        f"MAE: "
        f"{holdout_metrics['mae']:.2f}"
    )
    print(
        f"RMSE: "
        f"{holdout_metrics['rmse']:.2f}"
    )
    print(
        "Seasonal coverage: "
        f"{holdout_metrics['seasonal_coverage']:.2%}"
    )


if __name__ == "__main__":
    main()