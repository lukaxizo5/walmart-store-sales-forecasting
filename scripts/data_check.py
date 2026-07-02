from __future__ import annotations

from walmart_forecasting.data import (
    load_merged_data,
    load_raw_data,
)
from walmart_forecasting.paths import (
    RAW_DATA_DIR,
    create_output_directories,
)
from walmart_forecasting.validation import (
    competition_like_holdout,
    infer_forecast_horizon,
)


def main() -> None:
    create_output_directories()

    print(f"Reading data from: {RAW_DATA_DIR}")
    print()

    raw = load_raw_data()

    print("Raw files loaded successfully:")
    print(f"  train.csv:            {raw.train.shape}")
    print(f"  test.csv:             {raw.test.shape}")
    print(f"  features.csv:         {raw.features.shape}")
    print(f"  stores.csv:           {raw.stores.shape}")
    print(
        "  sampleSubmission.csv: "
        f"{raw.sample_submission.shape}"
    )
    print()

    merged = load_merged_data()

    print("Merged datasets:")
    print(f"  train: {merged.train.shape}")
    print(f"  test:  {merged.test.shape}")
    print()

    print("Training coverage:")
    print(
        f"  Start date: {merged.train['Date'].min().date()}"
    )
    print(
        f"  End date:   {merged.train['Date'].max().date()}"
    )
    print(
        f"  Unique weeks: "
        f"{merged.train['Date'].nunique()}"
    )
    print(
        f"  Stores: {merged.train['Store'].nunique()}"
    )
    print(
        f"  Departments: "
        f"{merged.train['Dept'].nunique()}"
    )
    print(
        "  Store-department series: "
        f"{merged.train[['Store', 'Dept']].drop_duplicates().shape[0]}"
    )
    print()

    forecast_horizon = infer_forecast_horizon(
        merged.test
    )

    print(f"Test forecast horizon: {forecast_horizon} weeks")

    split = competition_like_holdout(
        train_data=merged.train,
        test_data=merged.test,
    )

    print()
    print("Competition-like validation split:")
    print(f"  Training rows:   {len(split.train):,}")
    print(f"  Validation rows: {len(split.validation):,}")
    print(
        f"  Validation start: "
        f"{split.validation_start.date()}"
    )
    print(
        f"  Validation end:   "
        f"{split.validation_end.date()}"
    )
    print(
        f"  Validation weeks: "
        f"{split.validation_weeks}"
    )
    print()

    metadata_columns = [
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Type",
        "Size",
    ]

    print("Missing merged metadata:")
    print(
        merged.train[metadata_columns]
        .isna()
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    print()
    print("Data infrastructure check passed.")


if __name__ == "__main__":
    main()