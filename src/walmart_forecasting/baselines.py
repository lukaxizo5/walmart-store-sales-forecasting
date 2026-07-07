from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd


TARGET_COLUMN = "Weekly_Sales"
SERIES_COLUMNS = ["Store", "Dept"]


def lookup_pair_values(
    dataframe: pd.DataFrame,
    values: pd.Series,
) -> pd.Series:
    keys = pd.MultiIndex.from_frame(
        dataframe[SERIES_COLUMNS]
    )

    return pd.Series(
        values.reindex(keys).to_numpy(),
        index=dataframe.index,
        dtype=float,
    )


class SeasonalNaiveBaseline:
    def __init__(
        self,
        seasonal_lag: int = 52,
    ) -> None:
        self.seasonal_lag = seasonal_lag
        self.last_seasonal_coverage = 0.0

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> Self:
        history = (
            dataframe[
                [
                    *SERIES_COLUMNS,
                    "Date",
                    TARGET_COLUMN,
                ]
            ]
            .drop_duplicates(
                [
                    *SERIES_COLUMNS,
                    "Date",
                ]
            )
        )

        self.history = history.set_index(
            [
                *SERIES_COLUMNS,
                "Date",
            ]
        )[TARGET_COLUMN]

        self.pair_means = (
            dataframe.groupby(
                SERIES_COLUMNS
            )[TARGET_COLUMN]
            .mean()
        )

        self.department_means = (
            dataframe.groupby("Dept")[
                TARGET_COLUMN
            ]
            .mean()
        )

        self.store_means = (
            dataframe.groupby("Store")[
                TARGET_COLUMN
            ]
            .mean()
        )

        self.global_mean = float(
            dataframe[TARGET_COLUMN].mean()
        )

        return self

    def predict(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        lookup_rows = dataframe[
            [
                *SERIES_COLUMNS,
                "Date",
            ]
        ].copy()

        lookup_rows["Date"] = (
            lookup_rows["Date"]
            - pd.Timedelta(
                weeks=self.seasonal_lag
            )
        )

        seasonal_keys = pd.MultiIndex.from_frame(
            lookup_rows[
                [
                    *SERIES_COLUMNS,
                    "Date",
                ]
            ]
        )

        seasonal_values = pd.Series(
            self.history
            .reindex(seasonal_keys)
            .to_numpy(),
            index=dataframe.index,
            dtype=float,
        )

        self.last_seasonal_coverage = float(
            seasonal_values.notna().mean()
        )

        predictions = seasonal_values.copy()

        pair_values = lookup_pair_values(
            dataframe,
            self.pair_means,
        )

        predictions = predictions.fillna(
            pair_values
        )

        predictions = predictions.fillna(
            dataframe["Dept"].map(
                self.department_means
            )
        )

        predictions = predictions.fillna(
            dataframe["Store"].map(
                self.store_means
            )
        )

        predictions = predictions.fillna(
            self.global_mean
        )

        return predictions.to_numpy()