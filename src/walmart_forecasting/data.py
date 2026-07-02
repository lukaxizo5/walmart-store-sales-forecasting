from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype

from .paths import RAW_DATA_DIR


@dataclass(frozen=True)
class WalmartRawData:
    train: pd.DataFrame
    test: pd.DataFrame
    features: pd.DataFrame
    stores: pd.DataFrame
    sample_submission: pd.DataFrame


@dataclass(frozen=True)
class WalmartMergedData:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame


REQUIRED_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "features": "features.csv",
    "stores": "stores.csv",
    "sample_submission": "sampleSubmission.csv",
}

TRAIN_REQUIRED_COLUMNS = {
    "Store",
    "Dept",
    "Date",
    "Weekly_Sales",
    "IsHoliday",
}

TEST_REQUIRED_COLUMNS = {
    "Store",
    "Dept",
    "Date",
    "IsHoliday",
}

FEATURE_REQUIRED_COLUMNS = {
    "Store",
    "Date",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "IsHoliday",
}

STORE_REQUIRED_COLUMNS = {
    "Store",
    "Type",
    "Size",
}

SALES_KEY_COLUMNS = ["Store", "Dept", "Date"]
FEATURE_KEY_COLUMNS = ["Store", "Date", "IsHoliday"]


def _require_file(data_dir: Path, filename: str) -> Path:
    path = data_dir / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Extract the Kaggle CSV files into data/raw/."
        )

    return path


def _normalize_is_holiday(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    if "IsHoliday" not in dataframe.columns:
        return dataframe

    if is_bool_dtype(dataframe["IsHoliday"]):
        dataframe["IsHoliday"] = dataframe["IsHoliday"].astype(bool)
        return dataframe

    normalized = (
        dataframe["IsHoliday"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    converted = normalized.map(mapping)

    if converted.isna().any():
        invalid_values = sorted(
            dataframe.loc[converted.isna(), "IsHoliday"]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unexpected IsHoliday values: "
            f"{invalid_values}"
        )

    dataframe["IsHoliday"] = converted.astype(bool)

    return dataframe


def _read_dated_csv(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(
        path,
        parse_dates=["Date"],
    )

    return _normalize_is_holiday(dataframe)


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def _validate_duplicate_keys(
    dataframe: pd.DataFrame,
    key_columns: list[str],
    dataset_name: str,
) -> None:
    duplicate_mask = dataframe.duplicated(
        subset=key_columns,
        keep=False,
    )

    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count > 0:
        raise ValueError(
            f"{dataset_name} contains {duplicate_count} rows "
            f"with duplicate keys: {key_columns}"
        )


def validate_raw_data(data: WalmartRawData) -> None:
    _validate_columns(
        data.train,
        TRAIN_REQUIRED_COLUMNS,
        "train.csv",
    )
    _validate_columns(
        data.test,
        TEST_REQUIRED_COLUMNS,
        "test.csv",
    )
    _validate_columns(
        data.features,
        FEATURE_REQUIRED_COLUMNS,
        "features.csv",
    )
    _validate_columns(
        data.stores,
        STORE_REQUIRED_COLUMNS,
        "stores.csv",
    )

    _validate_duplicate_keys(
        data.train,
        SALES_KEY_COLUMNS,
        "train.csv",
    )
    _validate_duplicate_keys(
        data.test,
        SALES_KEY_COLUMNS,
        "test.csv",
    )
    _validate_duplicate_keys(
        data.features,
        FEATURE_KEY_COLUMNS,
        "features.csv",
    )
    _validate_duplicate_keys(
        data.stores,
        ["Store"],
        "stores.csv",
    )

    if data.train["Weekly_Sales"].isna().any():
        raise ValueError(
            "train.csv contains missing Weekly_Sales values."
        )

    if data.train["Date"].isna().any():
        raise ValueError("train.csv contains invalid dates.")

    if data.test["Date"].isna().any():
        raise ValueError("test.csv contains invalid dates.")

    if "Weekly_Sales" in data.test.columns:
        raise ValueError(
            "test.csv unexpectedly contains Weekly_Sales."
        )

    if len(data.sample_submission) != len(data.test):
        raise ValueError(
            "sampleSubmission.csv and test.csv have different "
            f"row counts: {len(data.sample_submission)} vs "
            f"{len(data.test)}."
        )


def load_raw_data(
    data_dir: str | Path = RAW_DATA_DIR,
) -> WalmartRawData:
    data_dir = Path(data_dir)

    paths = {
        name: _require_file(data_dir, filename)
        for name, filename in REQUIRED_FILES.items()
    }

    raw_data = WalmartRawData(
        train=_read_dated_csv(paths["train"]),
        test=_read_dated_csv(paths["test"]),
        features=_read_dated_csv(paths["features"]),
        stores=pd.read_csv(paths["stores"]),
        sample_submission=pd.read_csv(
            paths["sample_submission"]
        ),
    )

    validate_raw_data(raw_data)

    return raw_data


def merge_sales_with_metadata(
    sales: pd.DataFrame,
    features: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    original_row_count = len(sales)

    merged = sales.merge(
        features,
        on=FEATURE_KEY_COLUMNS,
        how="left",
        validate="many_to_one",
    )

    merged = merged.merge(
        stores,
        on="Store",
        how="left",
        validate="many_to_one",
    )

    if len(merged) != original_row_count:
        raise RuntimeError(
            "Merging changed the number of sales rows. "
            f"Before: {original_row_count}, after: {len(merged)}."
        )

    return merged


def load_merged_data(
    data_dir: str | Path = RAW_DATA_DIR,
) -> WalmartMergedData:
    raw = load_raw_data(data_dir)

    merged_train = merge_sales_with_metadata(
        sales=raw.train,
        features=raw.features,
        stores=raw.stores,
    )

    merged_test = merge_sales_with_metadata(
        sales=raw.test,
        features=raw.features,
        stores=raw.stores,
    )

    return WalmartMergedData(
        train=merged_train,
        test=merged_test,
        sample_submission=raw.sample_submission.copy(),
    )