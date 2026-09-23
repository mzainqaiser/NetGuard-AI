from pathlib import Path
from collections import Counter
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "normalized"
REPORTS_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LABEL NORMALIZATION MAPPING
# ============================================================

LABEL_MAPPING = {
    "BENIGN": "BENIGN",
    "DoS slowloris": "DoS Slowloris",
    "Web Attack \ufffd Brute Force": "Web Attack - Brute Force",
    "Web Attack \ufffd XSS": "Web Attack - XSS",
    "Web Attack \ufffd Sql Injection": "Web Attack - SQL Injection",
}


# ============================================================
# INITIALIZATION
# ============================================================

processed_files = sorted(INPUT_DIR.glob("*.csv"))

if not processed_files:
    raise FileNotFoundError(
        f"No CSV files found in input directory: {INPUT_DIR}"
    )

raw_label_counts = Counter()
normalized_label_counts = Counter()
mapping_counts = Counter()

total_rows = 0
total_files = 0


print("=" * 70)
print("TASK 10-C: LABEL NORMALIZATION")
print("=" * 70)

print(f"Input directory:  {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print()


# ============================================================
# PROCESS FILES ONE AT A TIME
# ============================================================

for file_path in processed_files:

    print(f"Processing: {file_path.name}")

    df = pd.read_csv(file_path, low_memory=False)

    if "Label" not in df.columns:
        raise ValueError(
            f"'Label' column not found in {file_path.name}"
        )

    original_rows = len(df)
    original_columns = len(df.columns)

    # Preserve the original label
    raw_labels = df["Label"].copy()

    # Store original labels in a new column
    df["Raw_Label"] = raw_labels

    # Normalize known labels
    normalized_labels = raw_labels.map(
        lambda label: LABEL_MAPPING.get(label, label)
    )

    # Verify that normalization did not create missing values
    if normalized_labels.isna().any():
        missing_count = normalized_labels.isna().sum()

        raise ValueError(
            f"Normalization created {missing_count} missing labels "
            f"in {file_path.name}"
        )

    # Replace Label with normalized labels
    df["Label"] = normalized_labels

    # Verify row count remains unchanged
    if len(df) != original_rows:
        raise ValueError(
            f"Row count changed in {file_path.name}"
        )

    # Verify column count increased by exactly one
    if len(df.columns) != original_columns + 1:
        raise ValueError(
            f"Unexpected column count in {file_path.name}"
        )

    # Collect statistics
    raw_label_counts.update(raw_labels)
    normalized_label_counts.update(normalized_labels)

    for raw_label, normalized_label in zip(
        raw_labels, normalized_labels
    ):
        mapping_counts[(raw_label, normalized_label)] += 1

    total_rows += len(df)
    total_files += 1

    # Save normalized dataset
    output_path = OUTPUT_DIR / file_path.name
    df.to_csv(output_path, index=False)

    print(
        f"  Rows: {original_rows:,} | "
        f"Columns: {original_columns} -> {len(df.columns)}"
    )

    print(f"  Saved: {output_path.name}")
    print()


# ============================================================
# CREATE LABEL MAPPING REPORT
# ============================================================

mapping_report = pd.DataFrame(
    [
        {
            "Raw_Label": raw_label,
            "Normalized_Label": normalized_label,
            "Record_Count": count,
        }
        for (raw_label, normalized_label), count
        in sorted(mapping_counts.items())
    ]
)

mapping_report_path = REPORTS_DIR / "label_normalization_mapping.csv"
mapping_report.to_csv(mapping_report_path, index=False)


# ============================================================
# CREATE NORMALIZED LABEL DISTRIBUTION REPORT
# ============================================================

distribution_report = pd.DataFrame(
    [
        {
            "Normalized_Label": label,
            "Record_Count": count,
            "Percentage": round(
                (count / total_rows) * 100, 4
            ),
        }
        for label, count
        in normalized_label_counts.most_common()
    ]
)

distribution_report_path = (
    REPORTS_DIR / "normalized_label_distribution.csv"
)

distribution_report.to_csv(
    distribution_report_path,
    index=False
)


# ============================================================
# CREATE SUMMARY REPORT
# ============================================================

summary_data = {
    "Total_Files_Processed": [total_files],
    "Total_Records": [total_rows],
    "Unique_Raw_Labels": [len(raw_label_counts)],
    "Unique_Normalized_Labels": [len(normalized_label_counts)],
    "Expected_Columns_Per_File": [80],
    "Raw_Data_Modified": ["No"],
    "Processed_Data_Overwritten": ["No"],
}

summary_report = pd.DataFrame(summary_data)

summary_report_path = (
    REPORTS_DIR / "label_normalization_summary.csv"
)

summary_report.to_csv(
    summary_report_path,
    index=False
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("=" * 70)
print("LABEL NORMALIZATION COMPLETED")
print("=" * 70)

print(f"Files processed:          {total_files}")
print(f"Total records:            {total_rows:,}")
print(f"Unique raw labels:        {len(raw_label_counts)}")
print(f"Unique normalized labels: {len(normalized_label_counts)}")
print()

print("Normalized Labels:")

for label, count in normalized_label_counts.most_common():
    print(f"  {label}: {count:,}")

print()
print("Reports created:")
print(f"  {mapping_report_path}")
print(f"  {distribution_report_path}")
print(f"  {summary_report_path}")

print()
print("Verification:")
print("  Original raw CSV files were not modified.")
print("  Existing processed CSV files were not overwritten.")
print("  Raw labels preserved in 'Raw_Label'.")
print("  Normalized labels stored in 'Label'.")
print("  Expected columns per file: 80")
print("=" * 70)