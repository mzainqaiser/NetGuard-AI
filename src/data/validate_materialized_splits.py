
"""
NetGuard AI — Post-Materialization Validation

Validates the materialized train/validation/test Parquet datasets
against the previously validated group-aware split assignments.

Checks:
1. All materialized files exist.
2. Expected record counts.
3. Expected 81-column schema.
4. Exactly 78 traffic features.
5. Label and Raw_Label columns exist.
6. Source_File provenance exists.
7. feature_group_id is NOT present in materialized ML data.
8. All 15 expected labels are present.
9. Feature-group IDs are recreated exactly from the 78 traffic features.
10. Every feature group maps to its validated split.
11. No feature-group overlap between train/validation/test.
12. Total unique feature groups = 2,521,725.
13. Reports are generated for label/source distributions and validation.

Designed for chunked processing with PyArrow to avoid loading
the complete Parquet datasets into RAM.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_ASSIGNMENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "group_metadata"
    / "feature_group_split_assignments.parquet"
)

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

SPLITS = {
    "train": SPLIT_DIR / "train" / "train.parquet",
    "validation": SPLIT_DIR / "validation" / "validation.parquet",
    "test": SPLIT_DIR / "test" / "test.parquet",
}

EXPECTED_RECORD_COUNTS = {
    "train": 1_981_516,
    "validation": 424_613,
    "test": 424_614,
}

EXPECTED_FEATURE_GROUPS = 2_521_725
EXPECTED_TOTAL_RECORDS = 2_830_743

EXPECTED_FEATURE_COUNT = 78
EXPECTED_MATERIALIZED_COLUMN_COUNT = 81

CHUNK_SIZE = 100_000

LABEL_COLUMN = "Label"
RAW_LABEL_COLUMN = "Raw_Label"
SOURCE_FILE_COLUMN = "Source_File"

FEATURE_GROUP_ID_COLUMN = "feature_group_id"
SPLIT_COLUMN = "split"

HASH_DIGEST_SIZE = 16

EXPECTED_LABELS = {
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS Slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Infiltration",
    "Web Attack - SQL Injection",
    "Heartbleed",
}


# ---------------------------------------------------------------------
# FEATURE-GROUP HASHING
# ---------------------------------------------------------------------

def get_feature_columns(columns: List[str]) -> List[str]:
    """
    Return the 78 traffic-feature columns.

    Label, Raw_Label, Source_File, and feature_group_id are excluded.
    """

    excluded = {
        LABEL_COLUMN,
        RAW_LABEL_COLUMN,
        SOURCE_FILE_COLUMN,
        FEATURE_GROUP_ID_COLUMN,
    }

    feature_columns = [
        column
        for column in columns
        if column not in excluded
    ]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} traffic features, "
            f"found {len(feature_columns)}."
        )

    return feature_columns


def normalize_value(value) -> str:
    """
    Match the exact normalization logic used by
    build_feature_groups.py.
    """

    if pd.isna(value):
        return "<NA>"

    if isinstance(value, float):
        return format(value, ".17g")

    return str(value)


def create_feature_group_id(
    row: pd.Series,
    feature_columns: List[str],
) -> str:
    """
    Recreate the deterministic BLAKE2b feature-group ID.

    IMPORTANT:
    This logic intentionally matches build_feature_groups.py.
    """

    hasher = hashlib.blake2b(
        digest_size=HASH_DIGEST_SIZE
    )

    for column in feature_columns:
        value = normalize_value(row[column])

        encoded = value.encode(
            "utf-8",
            errors="replace",
        )

        hasher.update(
            len(encoded).to_bytes(
                4,
                byteorder="big",
            )
        )

        hasher.update(encoded)

    return hasher.hexdigest()


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def load_split_assignments() -> Dict[str, str]:
    """
    Load validated feature-group -> split assignments.
    """

    print()
    print("Loading validated split assignments...")

    if not SPLIT_ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Split assignment file not found:\n"
            f"{SPLIT_ASSIGNMENTS_PATH}"
        )

    assignments_df = pd.read_parquet(
        SPLIT_ASSIGNMENTS_PATH
    )

    required_columns = {
        FEATURE_GROUP_ID_COLUMN,
        SPLIT_COLUMN,
    }

    missing = required_columns - set(
        assignments_df.columns
    )

    if missing:
        raise ValueError(
            f"Assignment file is missing columns: {sorted(missing)}"
        )

    if assignments_df[FEATURE_GROUP_ID_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate feature_group_id values found "
            "in split assignments."
        )

    assignments_df[SPLIT_COLUMN] = (
        assignments_df[SPLIT_COLUMN]
        .astype(str)
        .str.lower()
    )

    valid_splits = set(SPLITS.keys())

    unexpected_splits = (
        set(assignments_df[SPLIT_COLUMN].unique())
        - valid_splits
    )

    if unexpected_splits:
        raise ValueError(
            f"Unexpected split values: {sorted(unexpected_splits)}"
        )

    assignments = dict(
        zip(
            assignments_df[FEATURE_GROUP_ID_COLUMN],
            assignments_df[SPLIT_COLUMN],
        )
    )

    print(
        f"  Groups: {len(assignments):,}"
    )

    if len(assignments) != EXPECTED_FEATURE_GROUPS:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_GROUPS:,} groups, "
            f"found {len(assignments):,}."
        )

    return assignments


def validate_schema(
    parquet_path: Path,
) -> tuple[List[str], List[str]]:
    """
    Validate Parquet schema and return all columns + feature columns.
    """

    parquet_file = pq.ParquetFile(parquet_path)

    columns = parquet_file.schema_arrow.names

    print(f"  Columns: {len(columns)}")

    if len(columns) != EXPECTED_MATERIALIZED_COLUMN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MATERIALIZED_COLUMN_COUNT} columns, "
            f"found {len(columns)}."
        )

    required_columns = {
        LABEL_COLUMN,
        RAW_LABEL_COLUMN,
        SOURCE_FILE_COLUMN,
    }

    missing = required_columns - set(columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if FEATURE_GROUP_ID_COLUMN in columns:
        raise ValueError(
            "feature_group_id must NOT be present in "
            "the materialized ML dataset."
        )

    feature_columns = get_feature_columns(columns)

    print(
        f"  Features: {len(feature_columns)}"
    )

    return columns, feature_columns


# ---------------------------------------------------------------------
# SPLIT VALIDATION
# ---------------------------------------------------------------------

def validate_split(
    split_name: str,
    parquet_path: Path,
    assignments: Dict[str, str],
):
    """
    Validate one materialized split using PyArrow batches.
    """

    print()
    print("-" * 70)
    print(f"Validating {split_name.upper()}")
    print("-" * 70)

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Materialized file not found:\n{parquet_path}"
        )

    columns, feature_columns = validate_schema(
        parquet_path
    )

    parquet_file = pq.ParquetFile(
        parquet_path
    )

    actual_records = parquet_file.metadata.num_rows

    print(
        f"  Records: {actual_records:,}"
    )

    expected_records = EXPECTED_RECORD_COUNTS[
        split_name
    ]

    if actual_records != expected_records:
        raise ValueError(
            f"{split_name}: expected {expected_records:,} records, "
            f"found {actual_records:,}."
        )

    label_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}

    unique_groups: Set[str] = set()

    processed_records = 0

    batch_number = 0

    for batch in parquet_file.iter_batches(
        batch_size=CHUNK_SIZE
    ):
        batch_number += 1

        df = batch.to_pandas()

        processed_records += len(df)

        # -------------------------------------------------------------
        # Labels
        # -------------------------------------------------------------

        batch_labels = (
            df[LABEL_COLUMN]
            .astype(str)
            .value_counts()
        )

        for label, count in batch_labels.items():
            label_counts[label] = (
                label_counts.get(label, 0)
                + int(count)
            )

        # -------------------------------------------------------------
        # Source provenance
        # -------------------------------------------------------------

        batch_sources = (
            df[SOURCE_FILE_COLUMN]
            .astype(str)
            .value_counts()
        )

        for source, count in batch_sources.items():
            source_counts[source] = (
                source_counts.get(source, 0)
                + int(count)
            )

        # -------------------------------------------------------------
        # Recreate feature-group IDs
        # -------------------------------------------------------------

        for _, row in df.iterrows():
            feature_group_id = create_feature_group_id(
                row,
                feature_columns,
            )

            unique_groups.add(
                feature_group_id
            )

            assigned_split = assignments.get(
                feature_group_id
            )

            if assigned_split is None:
                raise ValueError(
                    f"Feature group {feature_group_id} "
                    f"was not found in validated split assignments."
                )

            if assigned_split != split_name:
                raise ValueError(
                    f"Feature-group leakage detected: "
                    f"group {feature_group_id} exists in "
                    f"{split_name}, but validated assignment "
                    f"is {assigned_split}."
                )

        if batch_number % 5 == 0:
            print(
                f"  Processed: {processed_records:,} records"
            )

    # -----------------------------------------------------------------
    # Final per-split checks
    # -----------------------------------------------------------------

    if processed_records != expected_records:
        raise ValueError(
            f"{split_name}: processed {processed_records:,} records, "
            f"expected {expected_records:,}."
        )

    unexpected_labels = (
        set(label_counts.keys())
        - EXPECTED_LABELS
    )

    if unexpected_labels:
        raise ValueError(
            f"{split_name}: unexpected labels found: "
            f"{sorted(unexpected_labels)}"
        )

    missing_labels = (
        EXPECTED_LABELS
        - set(label_counts.keys())
    )

    if missing_labels:
        raise ValueError(
            f"{split_name}: expected labels missing: "
            f"{sorted(missing_labels)}"
        )

    print(
        f"  Unique feature groups: "
        f"{len(unique_groups):,}"
    )

    print(
        f"  Labels present: "
        f"{len(label_counts)}/15"
    )

    print(
        f"  Source files: "
        f"{len(source_counts)}"
    )

    return {
        "records": processed_records,
        "unique_groups": unique_groups,
        "label_counts": label_counts,
        "source_counts": source_counts,
    }


# ---------------------------------------------------------------------
# REPORT GENERATION
# ---------------------------------------------------------------------

def write_reports(results: Dict[str, dict]) -> None:
    """
    Write validation reports.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Label distribution
    # -------------------------------------------------------------

    label_rows = []

    for label in sorted(EXPECTED_LABELS):
        row = {
            "Label": label,
            "train": results["train"]["label_counts"].get(
                label,
                0,
            ),
            "validation": results["validation"]["label_counts"].get(
                label,
                0,
            ),
            "test": results["test"]["label_counts"].get(
                label,
                0,
            ),
        }

        row["total"] = (
            row["train"]
            + row["validation"]
            + row["test"]
        )

        label_rows.append(row)

    label_df = pd.DataFrame(
        label_rows
    )

    label_report = (
        REPORTS_DIR
        / "materialized_split_label_distribution.csv"
    )

    label_df.to_csv(
        label_report,
        index=False,
    )

    # -------------------------------------------------------------
    # Source distribution
    # -------------------------------------------------------------

    all_sources = set()

    for split_result in results.values():
        all_sources.update(
            split_result["source_counts"].keys()
        )

    source_rows = []

    for source in sorted(all_sources):
        row = {
            "Source_File": source,
            "train": results["train"]["source_counts"].get(
                source,
                0,
            ),
            "validation": results["validation"]["source_counts"].get(
                source,
                0,
            ),
            "test": results["test"]["source_counts"].get(
                source,
                0,
            ),
        }

        row["total"] = (
            row["train"]
            + row["validation"]
            + row["test"]
        )

        source_rows.append(row)

    source_df = pd.DataFrame(
        source_rows
    )

    source_report = (
        REPORTS_DIR
        / "materialized_split_source_distribution.csv"
    )

    source_df.to_csv(
        source_report,
        index=False,
    )

    # -------------------------------------------------------------
    # Validation summary
    # -------------------------------------------------------------

    summary = {
        "status": "PASS",
        "expected_total_records": EXPECTED_TOTAL_RECORDS,
        "actual_total_records": sum(
            result["records"]
            for result in results.values()
        ),
        "expected_feature_groups": EXPECTED_FEATURE_GROUPS,
        "actual_unique_feature_groups": len(
            set().union(
                *[
                    result["unique_groups"]
                    for result in results.values()
                ]
            )
        ),
        "expected_materialized_columns": EXPECTED_MATERIALIZED_COLUMN_COUNT,
        "expected_feature_count": EXPECTED_FEATURE_COUNT,
        "splits": {},
    }

    for split_name, result in results.items():
        summary["splits"][split_name] = {
            "records": result["records"],
            "unique_feature_groups": len(
                result["unique_groups"]
            ),
            "labels": result["label_counts"],
            "source_file_count": len(
                result["source_counts"]
            ),
        }

    summary_path = (
        REPORTS_DIR
        / "materialized_split_validation_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("Reports written:")
    print(
        f"  {label_report.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"  {source_report.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"  {summary_path.relative_to(PROJECT_ROOT)}"
    )


# ---------------------------------------------------------------------
# MAIN VALIDATION
# ---------------------------------------------------------------------

def validate() -> None:
    """
    Execute complete post-materialization validation.
    """

    print("=" * 70)
    print(
        "NETGUARD AI — POST-MATERIALIZATION VALIDATION"
    )
    print("=" * 70)

    # -------------------------------------------------------------
    # Check files
    # -------------------------------------------------------------

    print()
    print("Checking materialized files...")

    for split_name, path in SPLITS.items():
        status = "FOUND" if path.exists() else "MISSING"

        print(
            f"  {split_name:<12} {status}"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing materialized split: {path}"
            )

    # -------------------------------------------------------------
    # Load assignments
    # -------------------------------------------------------------

    assignments = load_split_assignments()

    # -------------------------------------------------------------
    # Validate each split
    # -------------------------------------------------------------

    results = {}

    for split_name, parquet_path in SPLITS.items():
        results[split_name] = validate_split(
            split_name,
            parquet_path,
            assignments,
        )

    # -------------------------------------------------------------
    # Total record validation
    # -------------------------------------------------------------

    print()
    print("-" * 70)
    print("GLOBAL VALIDATION")
    print("-" * 70)

    total_records = sum(
        result["records"]
        for result in results.values()
    )

    print(
        f"  Total records: "
        f"{total_records:,}"
    )

    print(
        f"  Expected records: "
        f"{EXPECTED_TOTAL_RECORDS:,}"
    )

    if total_records != EXPECTED_TOTAL_RECORDS:
        raise ValueError(
            "Global record count mismatch."
        )

    # -------------------------------------------------------------
    # Feature-group overlap validation
    # -------------------------------------------------------------

    train_groups = results["train"]["unique_groups"]
    validation_groups = results["validation"]["unique_groups"]
    test_groups = results["test"]["unique_groups"]

    train_validation_overlap = (
        train_groups
        & validation_groups
    )

    train_test_overlap = (
        train_groups
        & test_groups
    )

    validation_test_overlap = (
        validation_groups
        & test_groups
    )

    print()
    print(
        "  Train ∩ Validation: "
        f"{len(train_validation_overlap):,}"
    )

    print(
        "  Train ∩ Test: "
        f"{len(train_test_overlap):,}"
    )

    print(
        "  Validation ∩ Test: "
        f"{len(validation_test_overlap):,}"
    )

    if train_validation_overlap:
        raise ValueError(
            "Feature-group overlap detected between "
            "train and validation."
        )

    if train_test_overlap:
        raise ValueError(
            "Feature-group overlap detected between "
            "train and test."
        )

    if validation_test_overlap:
        raise ValueError(
            "Feature-group overlap detected between "
            "validation and test."
        )

    # -------------------------------------------------------------
    # Global unique feature-group count
    # -------------------------------------------------------------

    all_groups = (
        train_groups
        | validation_groups
        | test_groups
    )

    print()
    print(
        f"  Unique feature groups: "
        f"{len(all_groups):,}"
    )

    print(
        f"  Expected feature groups: "
        f"{EXPECTED_FEATURE_GROUPS:,}"
    )

    if len(all_groups) != EXPECTED_FEATURE_GROUPS:
        raise ValueError(
            "Global feature-group count mismatch."
        )

    # -------------------------------------------------------------
    # Write reports
    # -------------------------------------------------------------

    write_reports(results)

    # -------------------------------------------------------------
    # Final status
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("STATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    validate()