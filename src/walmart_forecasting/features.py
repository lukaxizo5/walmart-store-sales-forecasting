from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


SERIES_COLUMNS = ["Store", "Dept"]
DATE_COLUMN = "Date"
TARGET_COLUMN = "Weekly_Sales"

MARKDOWN_COLUMNS = [
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
]

ECONOMIC_COLUMNS = [
    "CPI",
    "Unemployment",
]

MISSING_INDICATOR_COLUMNS = [
    *MARKDOWN_COLUMNS,
    *ECONOMIC_COLUMNS,
]


def add_calendar_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    result["year"] = result[DATE_COLUMN].dt.year
    result["month"] = result[DATE_COLUMN].dt.month
    result["quarter"] = result[DATE_COLUMN].dt.quarter
    result["week_of_year"] = (
        result[DATE_COLUMN]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    return result


def add_missing_indicators(
    dataframe: pd.DataFrame,
    columns: Iterable[str] = MISSING_INDICATOR_COLUMNS,
) -> pd.DataFrame:
    result = dataframe.copy()

    for column in columns:
        result[f"{column}_missing"] = (
            result[column].isna().astype(int)
        )

    return result


def add_basic_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = add_calendar_features(dataframe)
    result = add_missing_indicators(result)

    return result


def add_exact_lag_features(
    rows: pd.DataFrame,
    history: pd.DataFrame,
    lags: Iterable[int],
) -> pd.DataFrame:
    result = rows.copy()

    history_values = (
        history[
            [
                *SERIES_COLUMNS,
                DATE_COLUMN,
                TARGET_COLUMN,
            ]
        ]
        .drop_duplicates(
            [
                *SERIES_COLUMNS,
                DATE_COLUMN,
            ]
        )
    )

    for lag in lags:
        lagged = history_values.copy()

        lagged[DATE_COLUMN] = (
            lagged[DATE_COLUMN]
            + pd.Timedelta(weeks=lag)
        )

        lagged = lagged.rename(
            columns={
                TARGET_COLUMN: f"lag_{lag}"
            }
        )

        result = result.merge(
            lagged,
            on=[
                *SERIES_COLUMNS,
                DATE_COLUMN,
            ],
            how="left",
            validate="many_to_one",
            sort=False,
        )

    return result