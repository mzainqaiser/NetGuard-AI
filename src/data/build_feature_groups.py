"""
NetGuard AI - Task 13-B
Feature-Group ID / Provenance Pipeline

Purpose:
- Build deterministic feature-group IDs from the 78 traffic features.
- Preserve source-file and label provenance.
- Identify groups containing multiple labels.
- Create compact group-level metadata for the later
  train/validation/test split.
- Do NOT modify the source datasets.
"""

from pathlib import Path
from collections import defaultdict

import hashlib
import json

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NORMALIZED_DIR = PROJECT_ROOT / "data" / "normalized"

REPORTS_DIR = PROJECT_ROOT / "reports"

GROUP_METADATA_DIR = PROJECT_ROOT / "data" / "group_metadata"

GROUP_METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

LABEL_COLUMN = "Label"
RAW_LABEL_COLUMN = "Raw_Label"

EXPECTED_FEATURE_COUNT = 78

CHUNK_SIZE = 100_000

HASH_DIGEST_SIZE = 16

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def get_feature_columns(columns):
    """Return the 78 model feature columns."""

    excluded = {
        LABEL_COLUMN,
        RAW_LABEL_COLUMN,
    }

    feature_columns = [
        column
        for column in columns
        if column not in excluded
    ]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_FEATURE_COUNT} feature columns, "
            f"found {len(feature_columns)}."
        )

    return feature_columns


def normalize_value(value):
    """
    Convert feature values into deterministic text
    representation before hashing.
    """

    if pd.isna(value):
        return "<NA>"

    if isinstance(value, float):
        return format(value, ".17g")

    return str(value)


def create_feature_group_id(row, feature_columns):
    """
    Create a deterministic BLAKE2b fingerprint from
    the 78 traffic features.

    Label and Raw_Label are intentionally excluded.
    """

    hasher = hashlib.blake2b(
        digest_size=HASH_DIGEST_SIZE
    )

    for column in feature_columns:

        value = normalize_value(
            row[column]
        )

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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NETGUARD AI - TASK 13-B")
    print("FEATURE-GROUP ID / PROVENANCE PIPELINE")
    print("=" * 70)

    files = sorted(
        NORMALIZED_DIR.glob("*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No normalized CSV files found in:\n"
            f"{NORMALIZED_DIR}"
        )

    print(
        f"\nNormalized source files: {len(files)}"
    )

    # ========================================================
    # GROUP STORAGE
    # ========================================================

    # Maps feature-group ID to information about the group.
    groups = {}

    total_rows = 0

    # ========================================================
    # PROCESS FILES
    # ========================================================

    for file_path in files:

        print("\n" + "-" * 70)
        print(f"Processing: {file_path.name}")
        print("-" * 70)

        file_rows = 0

        for chunk_number, df in enumerate(
            pd.read_csv(
                file_path,
                chunksize=CHUNK_SIZE,
            ),
            start=1,
        ):

            feature_columns = get_feature_columns(
                df.columns
            )

            # ------------------------------------------------
            # Generate deterministic feature-group IDs
            # ------------------------------------------------

            group_ids = []

            for _, row in df.iterrows():

                group_id = create_feature_group_id(
                    row,
                    feature_columns,
                )

                group_ids.append(group_id)

                label = str(
                    row[LABEL_COLUMN]
                )

                raw_label = str(
                    row[RAW_LABEL_COLUMN]
                )

                if group_id not in groups:

                    groups[group_id] = {
                        "feature_group_id": group_id,
                        "row_count": 0,
                        "labels": set(),
                        "raw_labels": set(),
                        "source_files": set(),
                    }

                group = groups[group_id]

                group["row_count"] += 1

                group["labels"].add(label)

                group["raw_labels"].add(
                    raw_label
                )

                group["source_files"].add(
                    file_path.name
                )

            file_rows += len(df)
            total_rows += len(df)

            if chunk_number % 5 == 0:
                print(
                    f"  Processed chunk {chunk_number}: "
                    f"{file_rows:,} rows"
                )

        print(
            f"  File rows processed: "
            f"{file_rows:,}"
        )

    # ========================================================
    # CONVERT GROUPS TO REPORT RECORDS
    # ========================================================

    print("\n" + "=" * 70)
    print("BUILDING GROUP METADATA")
    print("=" * 70)

    metadata_records = []

    for group_id, group in groups.items():

        labels = sorted(
            group["labels"]
        )

        raw_labels = sorted(
            group["raw_labels"]
        )

        source_files = sorted(
            group["source_files"]
        )

        metadata_records.append(
            {
                "feature_group_id": group_id,
                "row_count": group["row_count"],
                "label_count": len(labels),
                "labels": " | ".join(labels),
                "raw_labels": " | ".join(
                    raw_labels
                ),
                "source_file_count": len(
                    source_files
                ),
                "source_files": " | ".join(
                    source_files
                ),
                "is_label_conflict": (
                    len(labels) > 1
                ),
            }
        )

    metadata_df = pd.DataFrame(
        metadata_records
    )

    metadata_df = metadata_df.sort_values(
        "feature_group_id"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    unique_groups = len(metadata_df)

    grouped_rows = int(
        metadata_df["row_count"].sum()
    )

    conflict_groups = int(
        metadata_df[
            "is_label_conflict"
        ].sum()
    )

    conflict_rows = int(
        metadata_df.loc[
            metadata_df["is_label_conflict"],
            "row_count",
        ].sum()
    )

    multi_file_groups = int(
        (
            metadata_df[
                "source_file_count"
            ] > 1
        ).sum()
    )

    print(
        f"\nTotal source records: "
        f"{total_rows:,}"
    )

    print(
        f"Unique feature groups: "
        f"{unique_groups:,}"
    )

    print(
        f"Rows represented by groups: "
        f"{grouped_rows:,}"
    )

    print(
        f"Multi-label feature groups: "
        f"{conflict_groups:,}"
    )

    print(
        f"Rows in multi-label groups: "
        f"{conflict_rows:,}"
    )

    print(
        f"Groups appearing in multiple files: "
        f"{multi_file_groups:,}"
    )

    # --------------------------------------------------------
    # Critical validation
    # --------------------------------------------------------

    if grouped_rows != total_rows:
        raise RuntimeError(
            "GROUPING VALIDATION FAILED: "
            "group row count does not equal source row count."
        )

    if conflict_groups != 697:
        raise RuntimeError(
            "CONFLICT VALIDATION FAILED: expected "
            f"697 verified conflict groups, found "
            f"{conflict_groups}."
        )

    if conflict_rows != 6666:
        raise RuntimeError(
            "CONFLICT ROW VALIDATION FAILED: expected "
            f"6,666 rows, found "
            f"{conflict_rows}."
        )

    # ========================================================
    # SAVE GROUP METADATA
    # ========================================================

    metadata_path = (
        GROUP_METADATA_DIR
        / "feature_group_metadata.csv"
    )

    metadata_df.to_csv(
        metadata_path,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "total_records": total_rows,
        "unique_feature_groups": unique_groups,
        "multi_label_groups": conflict_groups,
        "rows_in_multi_label_groups": conflict_rows,
        "multi_file_groups": multi_file_groups,
        "feature_count": EXPECTED_FEATURE_COUNT,
        "hash_algorithm": "BLAKE2b",
        "hash_digest_size_bytes": HASH_DIGEST_SIZE,
        "random_state": RANDOM_STATE,
        "source_directory": str(
            NORMALIZED_DIR
        ),
    }

    summary_path = (
        REPORTS_DIR
        / "feature_group_pipeline_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n" + "=" * 70)
    print("GROUPING VALIDATION")
    print("=" * 70)

    print(
        "✓ Source row count matches grouped row count"
    )

    print(
        "✓ 697 verified conflict groups reproduced"
    )

    print(
        "✓ 6,666 conflict rows reproduced"
    )

    print(
        "✓ 78 model features used for grouping"
    )

    print(
        "✓ Label and Raw_Label excluded from group ID"
    )

    print(
        "✓ BLAKE2b deterministic feature fingerprints created"
    )

    print("\nFiles created:")

    print(
        f"  {metadata_path}"
    )

    print(
        f"  {summary_path}"
    )

    print("\nSTATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()