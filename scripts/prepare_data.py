from walmart_forecasting.data import (
    load_merged_data,
    save_merged_data,
)


def main() -> None:
    data = load_merged_data()
    save_merged_data(data)

    print("Processed datasets saved successfully.")
    print(f"Train shape: {data.train.shape}")
    print(f"Test shape:  {data.test.shape}")


if __name__ == "__main__":
    main()