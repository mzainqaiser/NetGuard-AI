from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"F:\Projects\NetGuard-AI")

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "MachineLearningCSV"
    / "MachineLearningCVE"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMNS = [
    "Flow Bytes/s",
    "Flow Packets/s",
]


def analyze_file(file_path):
    """Analyze target feature distributions by label."""

    print(f"\nAnalyzing: {file_path.name}")

    df = pd.read_csv(file_path)

    # Standardize column names.
    df.columns = [column.strip() for column in df.columns]

    # Convert infinity to NaN.
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    rows = []

    for label, group in df.groupby("Label", sort=True):

        for column in TARGET_COLUMNS:

            values = group[column].dropna()

            if len(values) == 0:
                continue

            rows.append(
                {
                    "file": file_path.name,
                    "label": label,
                    "feature": column,
                    "count": int(values.count()),
                    "missing": int(group[column].isna().sum()),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )

    return rows


def main():

    print("=" * 70)
    print("NetGuard AI — Imputation Analysis")
    print("Phase 2, Task 09-D")
    print("=" * 70)

    files = sorted(DATASET_DIR.glob("*.csv"))

    if len(files) != 8:
        raise FileNotFoundError(
            f"Expected 8 CIC-IDS2017 CSV files, found {len(files)}."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for file_path in files:
        file_rows = analyze_file(file_path)
        all_rows.extend(file_rows)

    report = pd.DataFrame(all_rows)

    output_path = (
        REPORTS_DIR / "imputation_analysis_by_label.csv"
    )

    report.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("IMPUTATION ANALYSIS COMPLETED")
    print("=" * 70)

    print(f"\nReport created:")
    print(output_path)

    print("\nRaw CSV files were NOT modified.")


if __name__ == "__main__":
    main()