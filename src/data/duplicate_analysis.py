"""
NetGuard AI - Task 12-C
Duplicate Analysis & Data Leakage Assessment

Purpose:
- Analyze exact duplicate records in the normalized CIC-IDS2017 files.
- Analyze duplicate feature vectors with potentially different labels.
- Assess the risk of data leakage before train/validation/test splitting.
- Do NOT delete or modify any dataset records.
"""

from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NORMALIZED_DIR = PROJECT_ROOT / "data" / "normalized"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_FEATURE_COUNT = 78

LABEL_COLUMN = "Label"
RAW_LABEL_COLUMN = "Raw_Label"

CHUNK_SIZE = 100_000

EXPECTED_LABELS = {
    "BENIGN",
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS Slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "SSH-Patator",
    "Web Attack - Brute Force",
    "Web Attack - SQL Injection",
    "Web Attack - XSS",
}


# ============================================================
# HELPER
# ============================================================

def get_normalized_files():
    """Return all normalized dataset CSV files."""
    files = sorted(NORMALIZED_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No normalized CSV files found in: {NORMALIZED_DIR}"
        )

    return files


def calculate_duplicate_statistics(df):
    """Calculate exact duplicate statistics for one dataframe."""

    total_rows = len(df)

    exact_duplicate_mask = df.duplicated(keep=False)
    duplicate_rows = int(exact_duplicate_mask.sum())

    duplicate_groups = int(
        df.loc[exact_duplicate_mask]
        .drop_duplicates()
        .shape[0]
    )

    duplicate_extra_rows = duplicate_rows - duplicate_groups

    return {
        "total_rows": total_rows,
        "duplicate_rows": duplicate_rows,
        "duplicate_groups": duplicate_groups,
        "duplicate_extra_rows": duplicate_extra_rows,
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 70)
    print("NETGUARD AI - TASK 12-C")
    print("DUPLICATE ANALYSIS & DATA LEAKAGE ASSESSMENT")
    print("=" * 70)

    files = get_normalized_files()

    print(f"\nNormalized files found: {len(files)}")

    # --------------------------------------------------------
    # Overall counters
    # --------------------------------------------------------

    total_rows = 0
    total_duplicate_rows = 0
    total_duplicate_groups = 0

    file_results = []

    # Feature-only duplicate fingerprints.
    #
    # The feature hash excludes Label and Raw_Label.
    # This allows us to detect identical traffic features
    # associated with different labels.
    feature_hash_labels = defaultdict(set)
    feature_hash_counts = Counter()

    # Exact full-row fingerprints.
    exact_hash_counts = Counter()

    # --------------------------------------------------------
    # Analyze each file
    # --------------------------------------------------------

    for file_path in files:

        print("\n" + "-" * 70)
        print(f"Analyzing: {file_path.name}")
        print("-" * 70)

        file_rows = 0
        file_exact_duplicate_rows = 0

        # Read in chunks to keep memory usage reasonable.
        for chunk_number, df in enumerate(
            pd.read_csv(file_path, chunksize=CHUNK_SIZE),
            start=1,
        ):

            file_rows += len(df)

            # ------------------------------------------------
            # Validate expected structure
            # ------------------------------------------------

            if LABEL_COLUMN not in df.columns:
                raise ValueError(
                    f"{file_path.name}: missing '{LABEL_COLUMN}' column."
                )

            if RAW_LABEL_COLUMN not in df.columns:
                raise ValueError(
                    f"{file_path.name}: missing '{RAW_LABEL_COLUMN}' column."
                )

            feature_columns = [
                col
                for col in df.columns
                if col not in {LABEL_COLUMN, RAW_LABEL_COLUMN}
            ]

            if len(feature_columns) != EXPECTED_FEATURE_COUNT:
                raise ValueError(
                    f"{file_path.name}: expected "
                    f"{EXPECTED_FEATURE_COUNT} features, found "
                    f"{len(feature_columns)}."
                )

            # ------------------------------------------------
            # Exact duplicate rows inside the chunk
            #
            # This is only used as a quick local indicator.
            # Full-file duplicate statistics are calculated
            # using row fingerprints below.
            # ------------------------------------------------

            exact_hash = pd.util.hash_pandas_object(
                df,
                index=False,
            )

            exact_hash_counts.update(exact_hash.tolist())

            # ------------------------------------------------
            # Feature-only fingerprints
            # ------------------------------------------------

            feature_hash = pd.util.hash_pandas_object(
                df[feature_columns],
                index=False,
            )

            feature_hash_counts.update(feature_hash.tolist())

            labels = df[LABEL_COLUMN].astype(str).tolist()

            for feature_hash_value, label in zip(
                feature_hash.tolist(),
                labels,
            ):
                feature_hash_labels[feature_hash_value].add(label)

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if chunk_number % 5 == 0:
                print(
                    f"  Processed chunk {chunk_number}: "
                    f"{file_rows:,} rows"
                )

        # ----------------------------------------------------
        # File-level exact duplicate statistics
        # ----------------------------------------------------

        # Re-read the file in chunks and calculate exact
        # duplicate fingerprint counts specifically for this file.
        file_hash_counts = Counter()

        for df in pd.read_csv(file_path, chunksize=CHUNK_SIZE):

            file_hash = pd.util.hash_pandas_object(
                df,
                index=False,
            )

            file_hash_counts.update(file_hash.tolist())

        duplicate_groups = sum(
            1
            for count in file_hash_counts.values()
            if count > 1
        )

        duplicate_rows = sum(
            count
            for count in file_hash_counts.values()
            if count > 1
        )

        duplicate_extra_rows = sum(
            count - 1
            for count in file_hash_counts.values()
            if count > 1
        )

        total_rows += file_rows
        total_duplicate_rows += duplicate_rows
        total_duplicate_groups += duplicate_groups

        file_result = {
            "file": file_path.name,
            "rows": file_rows,
            "duplicate_groups": duplicate_groups,
            "duplicate_rows": duplicate_rows,
            "duplicate_extra_rows": duplicate_extra_rows,
            "duplicate_percentage": (
                duplicate_rows / file_rows * 100
                if file_rows
                else 0
            ),
        }

        file_results.append(file_result)

        print(f"  Rows: {file_rows:,}")
        print(f"  Duplicate groups: {duplicate_groups:,}")
        print(f"  Duplicate rows: {duplicate_rows:,}")
        print(f"  Extra duplicate rows: {duplicate_extra_rows:,}")

    # ========================================================
    # Feature-label conflict analysis
    # ========================================================

    feature_duplicate_groups = 0
    feature_duplicate_rows = 0
    conflicting_feature_groups = 0
    conflicting_feature_rows = 0

    conflict_rows = []

    for feature_hash_value, count in feature_hash_counts.items():

        if count <= 1:
            continue

        labels = feature_hash_labels[feature_hash_value]

        feature_duplicate_groups += 1
        feature_duplicate_rows += count

        if len(labels) > 1:

            conflicting_feature_groups += 1
            conflicting_feature_rows += count

            conflict_rows.append(
                {
                    "feature_hash": feature_hash_value,
                    "occurrences": count,
                    "label_count": len(labels),
                    "labels": " | ".join(sorted(labels)),
                }
            )

    # ========================================================
    # Overall duplicate summary
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "metric": "total_records",
                "value": total_rows,
            },
            {
                "metric": "exact_duplicate_groups",
                "value": total_duplicate_groups,
            },
            {
                "metric": "exact_duplicate_rows",
                "value": total_duplicate_rows,
            },
            {
                "metric": "exact_duplicate_extra_rows",
                "value": total_duplicate_rows
                - total_duplicate_groups,
            },
            {
                "metric": "exact_duplicate_percentage",
                "value": (
                    total_duplicate_rows / total_rows * 100
                    if total_rows
                    else 0
                ),
            },
            {
                "metric": "feature_duplicate_groups",
                "value": feature_duplicate_groups,
            },
            {
                "metric": "feature_duplicate_rows",
                "value": feature_duplicate_rows,
            },
            {
                "metric": "conflicting_feature_groups",
                "value": conflicting_feature_groups,
            },
            {
                "metric": "conflicting_feature_rows",
                "value": conflicting_feature_rows,
            },
            {
                "metric": "feature_conflict_percentage",
                "value": (
                    conflicting_feature_groups
                    / feature_duplicate_groups
                    * 100
                    if feature_duplicate_groups
                    else 0
                ),
            },
        ]
    )

    # ========================================================
    # Save reports
    # ========================================================

    summary_path = REPORTS_DIR / "duplicate_analysis_summary.csv"

    file_results_path = REPORTS_DIR / "duplicate_analysis_by_file.csv"

    conflicts_path = (
        REPORTS_DIR / "duplicate_feature_label_conflicts.csv"
    )

    summary.to_csv(summary_path, index=False)

    pd.DataFrame(file_results).to_csv(
        file_results_path,
        index=False,
    )

    pd.DataFrame(conflict_rows).to_csv(
        conflicts_path,
        index=False,
    )

    # ========================================================
    # Console report
    # ========================================================

    print("\n" + "=" * 70)
    print("DUPLICATE ANALYSIS RESULTS")
    print("=" * 70)

    print(f"Total records: {total_rows:,}")
    print(
        f"Exact duplicate groups: "
        f"{total_duplicate_groups:,}"
    )
    print(
        f"Exact duplicate rows: "
        f"{total_duplicate_rows:,}"
    )
    print(
        f"Extra duplicate rows: "
        f"{total_duplicate_rows - total_duplicate_groups:,}"
    )

    print(
        f"Exact duplicate percentage: "
        f"{total_duplicate_rows / total_rows * 100:.4f}%"
    )

    print("\nFeature-level duplicate analysis:")
    print(
        f"Feature duplicate groups: "
        f"{feature_duplicate_groups:,}"
    )
    print(
        f"Feature duplicate rows: "
        f"{feature_duplicate_rows:,}"
    )

    print("\nPotential label conflicts:")
    print(
        f"Conflicting feature groups: "
        f"{conflicting_feature_groups:,}"
    )
    print(
        f"Rows involved in conflicting groups: "
        f"{conflicting_feature_rows:,}"
    )

    print("\nReports created:")
    print(f"  {summary_path}")
    print(f"  {file_results_path}")
    print(f"  {conflicts_path}")

    # ========================================================
    # Assessment
    # ========================================================

    print("\n" + "=" * 70)
    print("DATA LEAKAGE ASSESSMENT")
    print("=" * 70)

    if total_duplicate_rows > 0:
        print(
            "WARNING: Duplicate records exist in the dataset."
        )
        print(
            "Random row-level splitting could place identical "
            "records in both training and validation/test sets."
        )
        print(
            "This can produce overly optimistic evaluation results."
        )
    else:
        print(
            "No exact duplicate records detected."
        )

    if conflicting_feature_groups > 0:
        print(
            "\nIMPORTANT: Some identical feature vectors have "
            "different labels."
        )
        print(
            "These cases require additional investigation before "
            "model training."
        )
    else:
        print(
            "\nNo feature-level label conflicts detected."
        )

    print(
        "\nRecommendation: do not delete duplicates automatically."
    )
    print(
        "The train/validation/test splitting strategy should "
        "account for duplicate or highly related records."
    )

    print("\nSTATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()