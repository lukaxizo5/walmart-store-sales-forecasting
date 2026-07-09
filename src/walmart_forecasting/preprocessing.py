from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import (
    ECONOMIC_COLUMNS,
    MARKDOWN_COLUMNS,
)


CATEGORICAL_COLUMNS = [
    "Store",
    "Dept",
    "Type",
]

BASE_NUMERIC_COLUMNS = [
    "Size",
    "IsHoliday",
    "Temperature",
    "Fuel_Price",
    "year",
    "month",
    "quarter",
    "week_of_year",
]

MARKDOWN_INDICATOR_COLUMNS = [
    "MarkDown1_missing",
    "MarkDown2_missing",
    "MarkDown3_missing",
    "MarkDown4_missing",
    "MarkDown5_missing",
]

ECONOMIC_INDICATOR_COLUMNS = [
    "CPI_missing",
    "Unemployment_missing",
]


def build_tree_preprocessor(
    extra_numeric_columns: Sequence[str] = (),
    include_economic: bool = True,
    markdown_strategy: str = "zero",
) -> ColumnTransformer:
    if markdown_strategy not in {
        "zero",
        "median",
    }:
        raise ValueError(
            "markdown_strategy must be "
            "'zero' or 'median'."
        )

    numeric_columns = [
        *BASE_NUMERIC_COLUMNS,
        *MARKDOWN_INDICATOR_COLUMNS,
        *extra_numeric_columns,
    ]

    if include_economic:
        numeric_columns.extend(
            ECONOMIC_INDICATOR_COLUMNS
        )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
        ]
    )

    if markdown_strategy == "zero":
        markdown_imputer = SimpleImputer(
            strategy="constant",
            fill_value=0,
            keep_empty_features=True,
        )
    else:
        markdown_imputer = SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    transformers = [
        (
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ),
        (
            "markdown",
            markdown_imputer,
            MARKDOWN_COLUMNS,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_COLUMNS,
        ),
    ]

    if include_economic:
        economic_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "economic",
                economic_pipeline,
                ECONOMIC_COLUMNS,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )