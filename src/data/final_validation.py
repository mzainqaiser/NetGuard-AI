from pathlib import Path
from collections import Counter
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "normalized"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPECTED STRUCTURE
# ============================================================

EXPECTED_FEATURE_COUNT = 78
EXPECTED_COLUMN_COUNT = 80

EXPECTED_LABELS = {
    "BENIGN",
    "DDoS",
    "PortScan",
    "Bot",
    "Infiltration",
    "Web Attack - Brute Force",
    "Web Attack - SQL Injection",
    "Web Attack - XSS",
    "FTP-Patator",
    "SSH-Patator",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS Slowloris",
    "Heartbleed",
}


# ============================================================
# INITIALIZATION
# ============================================================

files = sorted(INPUT_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError(
        f"No normalized CSV files found in: {INPUT_DIR}"
    )

print("=" * 70)
print("TASK 11: FINAL FEATURE & LABEL VALIDATION")
print("=" * 70)

print(f"Input directory: {INPUT_DIR}")
print(f"Files found: {len(files)}")
print()


# ============================================================
# REFERENCE COLUMN STRUCTURE
# ============================================================

reference_columns = None

file_results = []

global_labels = Counter()

total_rows = 0
total_missing = 0
total_infinite = 0
total_duplicate_rows = 0


# ============================================================
# PROCESS FILES ONE AT A TIME
# ============================================================

for file_path in files:

    print(f"Validating: {file_path.name}")

    df = pd.read_csv(file_path, low_memory=False)

    rows = len(df)
    columns = len(df.columns)

    # --------------------------------------------------------
    # Basic column checks
    # --------------------------------------------------------

    has_label = "Label" in df.columns
    has_raw_label = "Raw_Label" in df.columns

    duplicate_columns = df.columns[
        df.columns.duplicated()
    ].tolist()

    # --------------------------------------------------------
    # Column structure consistency
    # --------------------------------------------------------

    if reference_columns is None:
        reference_columns = list(df.columns)
        structure_match = True
    else:
        structure_match = (
            list(df.columns) == reference_columns
        )

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    if has_label and has_raw_label:
        feature_columns = [
            column
            for column in df.columns
            if column not in {"Label", "Raw_Label"}
        ]
    else:
        feature_columns = []

    feature_count = len(feature_columns)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = int(df.isna().sum().sum())

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    )

    infinite_values = int(
        numeric_columns.isin([float("inf"), float("-inf")])
        .sum()
        .sum()
    )

    # --------------------------------------------------------
    # Label validation
    # --------------------------------------------------------

    file_labels = set(df["Label"].dropna().unique()) \
        if has_label else set()

    unexpected_labels = sorted(
        file_labels - EXPECTED_LABELS
    )

    missing_expected_labels = sorted(
        EXPECTED_LABELS - file_labels
    )

    global_labels.update(
        df["Label"].dropna()
    )

    # --------------------------------------------------------
    # Raw label consistency
    # --------------------------------------------------------

    raw_label_mismatch = 0

    if has_label and has_raw_label:

        # Expected relationship between Raw_Label and Label
        raw_to_normalized = {
            "BENIGN": "BENIGN",
            "DoS slowloris": "DoS Slowloris",
            "Web Attack \ufffd Brute Force":
                "Web Attack - Brute Force",
            "Web Attack \ufffd XSS":
                "Web Attack - XSS",
            "Web Attack \ufffd Sql Injection":
                "Web Attack - SQL Injection",
        }

        expected_normalized = df["Raw_Label"].map(
            lambda label: raw_to_normalized.get(
                label, label
            )
        )

        raw_label_mismatch = int(
            (expected_normalized != df["Label"]).sum()
        )

    # --------------------------------------------------------
    # Duplicate row count
    # --------------------------------------------------------

    duplicate_rows = int(
        df.duplicated().sum()
    )

    # --------------------------------------------------------
    # Record totals
    # --------------------------------------------------------

    total_rows += rows
    total_missing += missing_values
    total_infinite += infinite_values
    total_duplicate_rows += duplicate_rows

    # --------------------------------------------------------
    # Overall file status
    # --------------------------------------------------------

    file_pass = (
        rows > 0
        and columns == EXPECTED_COLUMN_COUNT
        and feature_count == EXPECTED_FEATURE_COUNT
        and has_label
        and has_raw_label
        and not duplicate_columns
        and structure_match
        and missing_values == 0
        and infinite_values == 0
        and not unexpected_labels
        and raw_label_mismatch == 0
    )

    file_results.append(
        {
            "File": file_path.name,
            "Rows": rows,
            "Columns": columns,
            "Feature_Count": feature_count,
            "Missing_Values": missing_values,
            "Infinite_Values": infinite_values,
            "Duplicate_Columns": len(duplicate_columns),
            "Duplicate_Rows": duplicate_rows,
            "Structure_Match": structure_match,
            "Unexpected_Labels": len(unexpected_labels),
            "Raw_Label_Mismatches": raw_label_mismatch,
            "Validation_Status": (
                "PASS" if file_pass else "FAIL"
            ),
        }
    )

    print(
        f"  Rows: {rows:,} | "
        f"Columns: {columns} | "
        f"Features: {feature_count}"
    )

    print(
        f"  Missing: {missing_values:,} | "
        f"Infinite: {infinite_values:,} | "
        f"Duplicate rows: {duplicate_rows:,}"
    )

    print(
        f"  Validation: "
        f"{'PASS' if file_pass else 'FAIL'}"
    )

    print()


# ============================================================
# SAVE FILE VALIDATION REPORT
# ============================================================

validation_report = pd.DataFrame(file_results)

validation_report_path = (
    REPORTS_DIR / "final_feature_label_validation.csv"
)

validation_report.to_csv(
    validation_report_path,
    index=False
)


# ============================================================
# GLOBAL LABEL REPORT
# ============================================================

global_label_report = pd.DataFrame(
    [
        {
            "Label": label,
            "Record_Count": count,
            "Percentage": round(
                (count / total_rows) * 100,
                4
            ),
        }
        for label, count
        in global_labels.most_common()
    ]
)

global_label_report_path = (
    REPORTS_DIR / "final_label_validation_distribution.csv"
)

global_label_report.to_csv(
    global_label_report_path,
    index=False
)


# ============================================================
# GLOBAL VERIFICATION
# ============================================================

all_files_pass = all(
    result["Validation_Status"] == "PASS"
    for result in file_results
)

global_label_set = set(global_labels.keys())

label_set_match = (
    global_label_set == EXPECTED_LABELS
)

overall_pass = (
    all_files_pass
    and total_missing == 0
    and total_infinite == 0
    and label_set_match
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("=" * 70)
print("FINAL VALIDATION SUMMARY")
print("=" * 70)

print(f"Files validated:        {len(files)}")
print(f"Total records:          {total_rows:,}")
print(f"Total missing values:   {total_missing:,}")
print(f"Total infinite values:  {total_infinite:,}")
print(f"Total duplicate rows:   {total_duplicate_rows:,}")
print(f"Unique normalized labels: {len(global_label_set)}")
print()

print(
    f"Expected features per file: "
    f"{EXPECTED_FEATURE_COUNT}"
)

print(
    f"Expected columns per file: "
    f"{EXPECTED_COLUMN_COUNT}"
)

print(
    f"Label set validation: "
    f"{'PASS' if label_set_match else 'FAIL'}"
)

print()

print(
    "OVERALL VALIDATION: "
    f"{'PASS' if overall_pass else 'FAIL'}"
)

print()
print("Report created:")
print(f"  {validation_report_path}")
print(f"  {global_label_report_path}")

print("=" * 70)


# ============================================================
# STOP THE PIPELINE IF VALIDATION FAILS
# ============================================================

if not overall_pass:
    raise ValueError(
        "Final dataset validation FAILED. "
        "Review the generated reports before continuing."
    )