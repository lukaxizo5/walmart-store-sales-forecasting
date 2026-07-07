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

MISSING_INDICATOR_COLUMNS = [
    "MarkDown1_missing",
    "MarkDown2_missing",
    "MarkDown3_missing",
    "MarkDown4_missing",
    "MarkDown5_missing",
    "CPI_missing",
    "Unemployment_missing",
]


def build_tree_preprocessor(
    extra_numeric_columns: Sequence[str] = (),
    markdown_strategy: str = "zero",
) -> ColumnTransformer:
    if markdown_strategy == "zero":
        markdown_imputer = SimpleImputer(
            strategy="constant",
            fill_value=0,
        )
    elif markdown_strategy == "median":
        markdown_imputer = SimpleImputer(
            strategy="median",
        )
    else:
        raise ValueError(
            "markdown_strategy must be "
            "'zero' or 'median'."
        )

    numeric_columns = [
        *BASE_NUMERIC_COLUMNS,
        *MISSING_INDICATOR_COLUMNS,
        *extra_numeric_columns,
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
        ]
    )

    economic_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
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
                "economic",
                economic_pipeline,
                ECONOMIC_COLUMNS,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )