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


def analyze_file(file_path):
    """Analyze missing and infinite values in one CSV file."""

    print(f"\nAnalyzing: {file_path.name}")

    df = pd.read_csv(file_path)

    # Standardize column names.
    df.columns = [column.strip() for column in df.columns]

    # Convert infinite numeric values to NaN for analysis.
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Identify rows containing at least one missing value.
    rows_with_missing = df.isna().any(axis=1)

    affected_rows = df.loc[rows_with_missing]

    missing_cells = int(df.isna().sum().sum())
    affected_row_count = int(rows_with_missing.sum())

    print(f"Total rows: {len(df):,}")
    print(f"Missing/invalid cells: {missing_cells:,}")
    print(f"Rows affected: {affected_row_count:,}")

    # Missing values by column.
    column_missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    column_missing = column_missing[
        column_missing > 0
    ]

    # Missing values by label.
    label_missing = (
        affected_rows["Label"]
        .value_counts()
        .sort_values(ascending=False)
    )

    return {
        "file": file_path.name,
        "total_rows": len(df),
        "missing_cells": missing_cells,
        "affected_rows": affected_row_count,
        "column_missing": column_missing,
        "label_missing": label_missing,
    }


def main():
    print("=" * 70)
    print("NetGuard AI — Missing Value Analysis")
    print("Phase 2, Task 09-B")
    print("=" * 70)

    files = sorted(DATASET_DIR.glob("*.csv"))

    if len(files) != 8:
        raise FileNotFoundError(
            f"Expected 8 CIC-IDS2017 CSV files, found {len(files)}."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for file_path in files:
        result = analyze_file(file_path)
        all_results.append(result)

    # ---------------------------------------------------------
    # File-level summary
    # ---------------------------------------------------------

    summary_rows = []

    for result in all_results:
        summary_rows.append(
            {
                "file": result["file"],
                "total_rows": result["total_rows"],
                "missing_cells": result["missing_cells"],
                "affected_rows": result["affected_rows"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_path = (
        REPORTS_DIR / "missing_value_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Column-level report
    # ---------------------------------------------------------

    column_rows = []

    for result in all_results:
        for column, count in result["column_missing"].items():
            column_rows.append(
                {
                    "file": result["file"],
                    "column": column,
                    "missing_cells": int(count),
                }
            )

    column_df = pd.DataFrame(column_rows)

    column_path = (
        REPORTS_DIR / "missing_values_by_column.csv"
    )

    column_df.to_csv(
        column_path,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Label-level report
    # ---------------------------------------------------------

    label_rows = []

    for result in all_results:
        for label, count in result["label_missing"].items():
            label_rows.append(
                {
                    "file": result["file"],
                    "label": label,
                    "affected_rows": int(count),
                }
            )

    label_df = pd.DataFrame(label_rows)

    label_path = (
        REPORTS_DIR / "missing_values_by_label.csv"
    )

    label_df.to_csv(
        label_path,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("MISSING VALUE ANALYSIS COMPLETED")
    print("=" * 70)

    print(f"\nReports created:")
    print(f"1. {summary_path}")
    print(f"2. {column_path}")
    print(f"3. {label_path}")

    print("\nRaw CSV files were NOT modified.")


if __name__ == "__main__":
    main()