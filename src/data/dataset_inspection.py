from pathlib import Path

import pandas as pd


DATASET_PATH = Path(
    "data/raw/MachineLearningCSV/MachineLearningCVE/"
    "Monday-WorkingHours.pcap_ISCX.csv"
)


def inspect_dataset(file_path: Path) -> None:
    print("=" * 70)
    print("NetGuard AI - Dataset Inspection")
    print("=" * 70)

    print(f"\nDataset: {file_path}")

    if not file_path.exists():
        print("\nERROR: Dataset file not found.")
        return

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\n--- Column Names ---")
    for index, column in enumerate(df.columns, start=1):
        print(f"{index:02d}. {column}")

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())

    print("\n--- Label Distribution ---")
    print(df["Label"].value_counts(dropna=False))

    print("\n--- Memory Usage ---")
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"{memory_mb:.2f} MB")


if __name__ == "__main__":
    inspect_dataset(DATASET_PATH)