from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    validation_weeks: int


def infer_forecast_horizon(
    test_data: pd.DataFrame,
    date_column: str = "Date",
) -> int:
    if date_column not in test_data.columns:
        raise ValueError(
            f"Missing date column: {date_column}"
        )

    horizon = int(test_data[date_column].nunique())

    if horizon <= 0:
        raise ValueError(
            "Could not infer a positive forecast horizon."
        )

    return horizon


def chronological_holdout(
    data: pd.DataFrame,
    validation_weeks: int,
    date_column: str = "Date",
) -> ChronologicalSplit:
    if date_column not in data.columns:
        raise ValueError(
            f"Missing date column: {date_column}"
        )

    if validation_weeks <= 0:
        raise ValueError(
            "validation_weeks must be positive."
        )

    unique_dates = (
        pd.Series(pd.to_datetime(data[date_column]))
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if validation_weeks >= len(unique_dates):
        raise ValueError(
            "Validation horizon must be smaller than the number "
            f"of available dates. Available dates: "
            f"{len(unique_dates)}, requested: {validation_weeks}."
        )

    validation_dates = unique_dates.iloc[-validation_weeks:]
    validation_start = pd.Timestamp(validation_dates.iloc[0])
    validation_end = pd.Timestamp(validation_dates.iloc[-1])

    training = data.loc[
        data[date_column] < validation_start
    ].copy()

    validation = data.loc[
        data[date_column].between(
            validation_start,
            validation_end,
            inclusive="both",
        )
    ].copy()

    if training.empty:
        raise ValueError(
            "Chronological split produced an empty training set."
        )

    if validation.empty:
        raise ValueError(
            "Chronological split produced an empty validation set."
        )

    if (
        training[date_column].max()
        >= validation[date_column].min()
    ):
        raise RuntimeError(
            "Temporal leakage detected: training dates overlap "
            "with validation dates."
        )

    return ChronologicalSplit(
        train=training,
        validation=validation,
        validation_start=validation_start,
        validation_end=validation_end,
        validation_weeks=validation_weeks,
    )


def competition_like_holdout(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    date_column: str = "Date",
) -> ChronologicalSplit:
    validation_weeks = infer_forecast_horizon(
        test_data,
        date_column=date_column,
    )

    return chronological_holdout(
        data=train_data,
        validation_weeks=validation_weeks,
        date_column=date_column,
    )