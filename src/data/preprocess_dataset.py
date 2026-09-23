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

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COLUMNS = [
    "Flow Bytes/s",
    "Flow Packets/s",
]


def standardize_columns(columns):
    """Remove leading/trailing whitespace from column names."""
    return [column.strip() for column in columns]


def process_file(file_path):
    """Preprocess one CSV and save the cleaned version."""

    print(f"\nProcessing: {file_path.name}")

    # ---------------------------------------------------------
    # Load raw CSV
    # ---------------------------------------------------------

    df = pd.read_csv(file_path)

    original_row_count = len(df)
    original_column_count = len(df.columns)

    # ---------------------------------------------------------
    # Standardize column names
    # ---------------------------------------------------------

    df.columns = standardize_columns(df.columns)

    if "Label" not in df.columns:
        raise ValueError(
            f"'Label' column not found in {file_path.name}"
        )

    # ---------------------------------------------------------
    # Identify numeric columns
    # ---------------------------------------------------------

    feature_columns = [
        column for column in df.columns
        if column != "Label"
    ]

    numeric_columns = df[feature_columns].select_dtypes(
        include=np.number
    ).columns

    # ---------------------------------------------------------
    # Convert infinite values to NaN
    # ---------------------------------------------------------

    infinite_before = int(
        np.isinf(df[numeric_columns]).sum().sum()
    )

    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ---------------------------------------------------------
    # Count missing values before imputation
    # ---------------------------------------------------------

    missing_before = int(
        df[TARGET_COLUMNS].isna().sum().sum()
    )

    affected_rows_before = int(
        df[TARGET_COLUMNS].isna().any(axis=1).sum()
    )

    # ---------------------------------------------------------
    # Group-wise median imputation
    #
    # Median is calculated separately for each Label group.
    # This prevents values from one traffic class being used
    # to impute another class.
    # ---------------------------------------------------------

    for column in TARGET_COLUMNS:

        group_medians = df.groupby("Label")[column].transform(
            "median"
        )

        df[column] = df[column].fillna(group_medians)

    # ---------------------------------------------------------
    # Verify target columns after imputation
    # ---------------------------------------------------------

    missing_after = int(
        df[TARGET_COLUMNS].isna().sum().sum()
    )

    affected_rows_after = int(
        df[TARGET_COLUMNS].isna().any(axis=1).sum()
    )

    if missing_after != 0:
        raise ValueError(
            f"Imputation failed in {file_path.name}. "
            f"Remaining missing target values: {missing_after}"
        )

    # ---------------------------------------------------------
    # Verify row/column counts
    # ---------------------------------------------------------

    if len(df) != original_row_count:
        raise ValueError(
            f"Row count changed in {file_path.name}."
        )

    if len(df.columns) != original_column_count:
        raise ValueError(
            f"Column count changed in {file_path.name}."
        )

    # ---------------------------------------------------------
    # Save processed file
    # ---------------------------------------------------------

    output_path = PROCESSED_DIR / file_path.name

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print(f"Rows: {original_row_count:,}")
    print(f"Columns: {original_column_count}")
    print(f"Infinite values converted to NaN: {infinite_before:,}")
    print(f"Missing target values before imputation: {missing_before:,}")
    print(f"Affected rows before imputation: {affected_rows_before:,}")
    print(f"Missing target values after imputation: {missing_after:,}")
    print(f"Affected rows after imputation: {affected_rows_after:,}")
    print(f"Saved: {output_path}")

    return {
        "file": file_path.name,
        "rows": original_row_count,
        "columns": original_column_count,
        "infinite_converted": infinite_before,
        "missing_before": missing_before,
        "affected_rows_before": affected_rows_before,
        "missing_after": missing_after,
        "affected_rows_after": affected_rows_after,
    }


def main():
    print("=" * 70)
    print("NetGuard AI — Safe Median Imputation")
    print("Phase 2, Task 09-E")
    print("=" * 70)

    files = sorted(DATASET_DIR.glob("*.csv"))

    if len(files) != 8:
        raise FileNotFoundError(
            f"Expected 8 CIC-IDS2017 CSV files, found {len(files)}."
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nDataset files found: {len(files)}")
    print(f"Processed output directory: {PROCESSED_DIR}")

    results = []

    for file_path in files:
        result = process_file(file_path)
        results.append(result)

    # ---------------------------------------------------------
    # Create preprocessing summary
    # ---------------------------------------------------------

    summary = pd.DataFrame(results)

    summary_path = (
        PROJECT_ROOT
        / "reports"
        / "preprocessing_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Final verification
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TASK 09-E COMPLETED")
    print("=" * 70)

    print("\nAll eight files were processed successfully.")
    print("Group-wise median imputation was applied.")
    print("No rows were deleted.")
    print("Raw CSV files were NOT modified.")
    print(f"Preprocessing summary: {summary_path}")


if __name__ == "__main__":
    main()