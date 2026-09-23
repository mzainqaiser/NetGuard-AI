
"""
NetGuard AI - Task 13-D.1
Strict Group Split Assignment Validation

Validates the generated group-aware train/validation/test assignment
before any actual datasets are materialized.

Assignment file:
    feature_group_id + split information

Group metadata file:
    feature_group_id
    row_count
    label_count
    labels
    raw_labels
    source_file_count
    source_files
    is_label_conflict

Checks:
1. Every feature group has exactly one split.
2. No duplicate group assignments.
3. All expected groups are accounted for.
4. All expected records are accounted for exactly once.
5. All 15 labels are present.
6. Label distribution is measured across splits.
7. Rare classes are explicitly inspected.
8. All 697 conflicting groups remain intact.
9. All 34,175 cross-source groups remain intact.
10. Train/validation/test group intersections are zero.
11. Deterministic assignment fingerprint is generated.

This script does NOT create train/validation/test datasets.
"""

from pathlib import Path
import hashlib
import json

import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ASSIGNMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "group_metadata"
    / "feature_group_split_assignments.parquet"
)

GROUP_METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "group_metadata"
    / "feature_group_metadata.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_SUMMARY = (
    REPORT_DIR / "group_split_validation_summary.json"
)

OUTPUT_LABEL_DISTRIBUTION = (
    REPORT_DIR / "group_split_label_distribution.csv"
)

OUTPUT_GROUP_VALIDATION = (
    REPORT_DIR / "group_split_group_validation.csv"
)

OUTPUT_CONFLICT_VALIDATION = (
    REPORT_DIR / "group_split_conflict_validation.csv"
)

OUTPUT_SOURCE_VALIDATION = (
    REPORT_DIR / "group_split_source_validation.csv"
)

RANDOM_STATE = 42

EXPECTED_GROUPS = 2_521_725
EXPECTED_RECORDS = 2_830_743

EXPECTED_CONFLICT_GROUPS = 697
EXPECTED_CONFLICT_ROWS = 6_666

EXPECTED_CROSS_SOURCE_GROUPS = 34_175

TARGET_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

SPLITS = [
    "train",
    "validation",
    "test",
]

EXPECTED_LABELS = [
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
]

RARE_CLASS_THRESHOLD = 1_000


# ============================================================================
# HELPERS
# ============================================================================

def fail(message: str):
    print("\nSTATUS: FAIL")
    print(f"ERROR: {message}")
    raise SystemExit(1)


def stable_hash_dataframe(
    df: pd.DataFrame,
    columns: list[str],
) -> str:
    """
    Create deterministic SHA-256 fingerprint from selected columns.
    """

    ordered = (
        df[columns]
        .sort_values(
            by=columns,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    csv_bytes = ordered.to_csv(
        index=False,
        lineterminator="\n",
    ).encode("utf-8")

    return hashlib.sha256(csv_bytes).hexdigest()


def parse_label_set(value) -> list[str]:
    """
    Parse the pipe-separated normalized label representation.
    """

    if pd.isna(value):
        return []

    labels = [
        item.strip()
        for item in str(value).split(" | ")
        if item.strip()
    ]

    return labels


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("NETGUARD AI - TASK 13-D.1")
    print("STRICT SPLIT ASSIGNMENT VALIDATION")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # 1. INPUT FILES
    # ------------------------------------------------------------------------

    print("\n[1/10] Checking input files...")

    if not ASSIGNMENT_FILE.exists():
        fail(
            f"Assignment file not found:\n{ASSIGNMENT_FILE}"
        )

    if not GROUP_METADATA_FILE.exists():
        fail(
            f"Group metadata file not found:\n{GROUP_METADATA_FILE}"
        )

    print("✓ Assignment file found:")
    print(f"  {ASSIGNMENT_FILE}")

    print("✓ Group metadata file found:")
    print(f"  {GROUP_METADATA_FILE}")

    # ------------------------------------------------------------------------
    # 2. LOAD ASSIGNMENT
    # ------------------------------------------------------------------------

    print("\n[2/10] Loading split assignments...")

    assignments = pd.read_parquet(
        ASSIGNMENT_FILE
    )

    required_assignment_columns = {
        "feature_group_id",
        "split",
    }

    missing_assignment_columns = (
        required_assignment_columns
        - set(assignments.columns)
    )

    if missing_assignment_columns:
        fail(
            "Assignment file is missing required columns: "
            f"{sorted(missing_assignment_columns)}"
        )

    print(
        f"Assignment rows: "
        f"{len(assignments):,}"
    )

    print(
        f"Assignment columns: "
        f"{len(assignments.columns)}"
    )

    # ------------------------------------------------------------------------
    # 3. GROUP INTEGRITY
    # ------------------------------------------------------------------------

    print("\n[3/10] Validating group integrity...")

    duplicate_group_ids = (
        assignments["feature_group_id"]
        .duplicated()
        .sum()
    )

    if duplicate_group_ids != 0:
        fail(
            "Duplicate feature_group_id assignments found: "
            f"{duplicate_group_ids:,}"
        )

    unique_groups = (
        assignments["feature_group_id"]
        .nunique()
    )

    if unique_groups != EXPECTED_GROUPS:
        fail(
            f"Expected {EXPECTED_GROUPS:,} unique groups, "
            f"found {unique_groups:,}"
        )

    if len(assignments) != EXPECTED_GROUPS:
        fail(
            f"Expected {EXPECTED_GROUPS:,} assignment rows, "
            f"found {len(assignments):,}"
        )

    null_split_count = (
        assignments["split"]
        .isna()
        .sum()
    )

    if null_split_count != 0:
        fail(
            f"Null split assignments found: "
            f"{null_split_count:,}"
        )

    invalid_split_values = sorted(
        set(assignments["split"].unique())
        - set(SPLITS)
    )

    if invalid_split_values:
        fail(
            "Invalid split values found: "
            f"{invalid_split_values}"
        )

    print(
        f"✓ Unique feature groups: "
        f"{unique_groups:,}"
    )

    print(
        f"✓ Duplicate group IDs: "
        f"{duplicate_group_ids:,}"
    )

    print(
        "✓ Every group has exactly one split assignment"
    )

    # ------------------------------------------------------------------------
    # 4. LOAD METADATA + MERGE
    # ------------------------------------------------------------------------

    print(
        "\n[4/10] Loading group metadata "
        "and validating record allocation..."
    )

    metadata = pd.read_csv(
        GROUP_METADATA_FILE
    )

    # Actual metadata schema:
    #
    # feature_group_id
    # row_count
    # label_count
    # labels
    # raw_labels
    # source_file_count
    # source_files
    # is_label_conflict

    required_metadata_columns = {
        "feature_group_id",
        "row_count",
        "label_count",
        "labels",
        "raw_labels",
        "source_file_count",
        "source_files",
        "is_label_conflict",
    }

    missing_metadata_columns = (
        required_metadata_columns
        - set(metadata.columns)
    )

    if missing_metadata_columns:
        fail(
            "Group metadata is missing required columns: "
            f"{sorted(missing_metadata_columns)}"
        )

    if len(metadata) != EXPECTED_GROUPS:
        fail(
            f"Expected {EXPECTED_GROUPS:,} metadata groups, "
            f"found {len(metadata):,}"
        )

    print(
        f"✓ Metadata groups: "
        f"{len(metadata):,}"
    )

    # Rename only internally.
    #
    # This does NOT modify the original metadata CSV.
    metadata = metadata.rename(
        columns={
            "row_count": "group_size",
            "labels": "label_set",
        }
    )

    # ------------------------------------------------------------------------
    # Verify assignment IDs == metadata IDs
    # ------------------------------------------------------------------------

    assignment_ids = set(
        assignments["feature_group_id"]
    )

    metadata_ids = set(
        metadata["feature_group_id"]
    )

    missing_in_assignment = (
        metadata_ids - assignment_ids
    )

    missing_in_metadata = (
        assignment_ids - metadata_ids
    )

    if missing_in_assignment:
        fail(
            "Metadata groups missing from assignment: "
            f"{len(missing_in_assignment):,}"
        )

    if missing_in_metadata:
        fail(
            "Assignment groups missing from metadata: "
            f"{len(missing_in_metadata):,}"
        )

    print(
        "✓ Assignment and metadata contain "
        "identical group IDs"
    )

    # ------------------------------------------------------------------------
    # Merge assignment + metadata
    # ------------------------------------------------------------------------

    merged = metadata.merge(
        assignments[
            [
                "feature_group_id",
                "split",
            ]
        ],
        on="feature_group_id",
        how="left",
        validate="one_to_one",
    )

    if merged["split"].isna().any():
        fail(
            "Some metadata groups have no split assignment"
        )

    if merged["group_size"].isna().any():
        fail(
            "Some groups have missing group_size values"
        )

    # ------------------------------------------------------------------------
    # Verify metadata record total
    # ------------------------------------------------------------------------

    metadata_record_total = int(
        merged["group_size"].sum()
    )

    if metadata_record_total != EXPECTED_RECORDS:
        fail(
            f"Metadata group sizes account for "
            f"{metadata_record_total:,} records, "
            f"expected {EXPECTED_RECORDS:,}"
        )

    # ------------------------------------------------------------------------
    # Calculate split group counts
    # ------------------------------------------------------------------------

    split_group_counts = (
        merged
        .groupby("split")["feature_group_id"]
        .nunique()
        .reindex(
            SPLITS,
            fill_value=0,
        )
    )

    # ------------------------------------------------------------------------
    # Calculate split record counts
    # ------------------------------------------------------------------------

    split_record_counts = (
        merged
        .groupby("split")["group_size"]
        .sum()
        .reindex(
            SPLITS,
            fill_value=0,
        )
    )

    split_record_total = int(
        split_record_counts.sum()
    )

    if split_record_total != EXPECTED_RECORDS:
        fail(
            f"Split record totals equal "
            f"{split_record_total:,}, "
            f"expected {EXPECTED_RECORDS:,}"
        )

    # ------------------------------------------------------------------------
    # Print split allocation
    # ------------------------------------------------------------------------

    split_rows = []

    print("\nSplit allocation:")

    for split in SPLITS:

        group_count = int(
            split_group_counts[split]
        )

        record_count = int(
            split_record_counts[split]
        )

        actual_ratio = (
            record_count
            / EXPECTED_RECORDS
        )

        target_ratio = (
            TARGET_RATIOS[split]
        )

        difference = (
            actual_ratio
            - target_ratio
        )

        split_rows.append(
            {
                "split": split,
                "groups": group_count,
                "records": record_count,
                "actual_ratio": actual_ratio,
                "target_ratio": target_ratio,
                "difference": difference,
            }
        )

        print(
            f"{split:12s}: "
            f"{group_count:>10,} groups | "
            f"{record_count:>10,} records | "
            f"{actual_ratio * 100:>8.4f}% | "
            f"target={target_ratio * 100:.2f}%"
        )

    print(
        f"\n✓ Metadata record total: "
        f"{metadata_record_total:,}"
    )

    print(
        f"✓ Split record total: "
        f"{split_record_total:,}"
    )

    print(
        "✓ All records accounted for exactly once"
    )

    # ------------------------------------------------------------------------
    # 5. LABEL DISTRIBUTION
    # ------------------------------------------------------------------------

    print(
        "\n[5/10] Validating label distribution..."
    )

    observed_labels = set()

    for value in merged["label_set"]:

        for label in parse_label_set(value):

            observed_labels.add(label)

    missing_labels = (
        set(EXPECTED_LABELS)
        - observed_labels
    )

    unexpected_labels = (
        observed_labels
        - set(EXPECTED_LABELS)
    )

    if missing_labels:
        fail(
            "Expected labels missing from split metadata: "
            f"{sorted(missing_labels)}"
        )

    if unexpected_labels:
        fail(
            "Unexpected labels found: "
            f"{sorted(unexpected_labels)}"
        )

    print(
        f"✓ Expected labels present: "
        f"{len(EXPECTED_LABELS)}"
    )

    print(
        "✓ No unexpected labels found"
    )

    # ------------------------------------------------------------------------
    # Build label distribution
    # ------------------------------------------------------------------------

    label_rows = []

    for label in EXPECTED_LABELS:

        label_mask = merged["label_set"].apply(
            lambda value:
                label in parse_label_set(value)
        )

        label_groups = merged[
            label_mask
        ]

        total_groups = len(
            label_groups
        )

        total_records = int(
            label_groups["group_size"].sum()
        )

        row = {
            "label": label,
            "total_groups": total_groups,
            "total_records": total_records,
        }

        for split in SPLITS:

            split_groups = label_groups[
                label_groups["split"] == split
            ]

            split_group_count = len(
                split_groups
            )

            split_record_count = int(
                split_groups["group_size"].sum()
            )

            if total_records > 0:
                percentage = (
                    split_record_count
                    / total_records
                )
            else:
                percentage = 0.0

            row[
                f"{split}_groups"
            ] = split_group_count

            row[
                f"{split}_records"
            ] = split_record_count

            row[
                f"{split}_percentage"
            ] = percentage

        label_rows.append(row)

    label_distribution = pd.DataFrame(
        label_rows
    )

    print("\nLabel distribution:")

    for _, row in label_distribution.iterrows():

        print(
            f"{row['label']:30s} | "
            f"total={int(row['total_records']):>10,} | "
            f"train={int(row['train_records']):>10,} "
            f"({row['train_percentage'] * 100:6.2f}%) | "
            f"val={int(row['validation_records']):>10,} "
            f"({row['validation_percentage'] * 100:6.2f}%) | "
            f"test={int(row['test_records']):>10,} "
            f"({row['test_percentage'] * 100:6.2f}%)"
        )

    # ------------------------------------------------------------------------
    # Rare classes
    # ------------------------------------------------------------------------

    print(
        "\nRare classes "
        f"(<= {RARE_CLASS_THRESHOLD:,} records):"
    )

    rare_distribution = label_distribution[
        label_distribution["total_records"]
        <= RARE_CLASS_THRESHOLD
    ]

    for _, row in rare_distribution.iterrows():

        print(
            f"  {row['label']:30s} "
            f"train={int(row['train_records']):>5,} | "
            f"val={int(row['validation_records']):>5,} | "
            f"test={int(row['test_records']):>5,}"
        )

    # ------------------------------------------------------------------------
    # 6. CONFLICT GROUPS
    # ------------------------------------------------------------------------

    print(
        "\n[6/10] Validating conflict-group integrity..."
    )

    # Use the authoritative metadata field.
    conflict_mask = (
        merged["is_label_conflict"]
        .fillna(False)
        .astype(bool)
    )

    conflict_groups = merged[
        conflict_mask
    ].copy()

    conflict_group_count = len(
        conflict_groups
    )

    conflict_record_count = int(
        conflict_groups["group_size"].sum()
    )

    if conflict_group_count != EXPECTED_CONFLICT_GROUPS:
        fail(
            f"Expected {EXPECTED_CONFLICT_GROUPS:,} "
            f"conflict groups, found "
            f"{conflict_group_count:,}"
        )

    if conflict_record_count != EXPECTED_CONFLICT_ROWS:
        fail(
            f"Expected {EXPECTED_CONFLICT_ROWS:,} "
            f"conflict records, found "
            f"{conflict_record_count:,}"
        )

    conflict_split_distribution = (
        conflict_groups
        .groupby("split")["group_size"]
        .agg(["count", "sum"])
        .reindex(
            SPLITS,
            fill_value=0,
        )
    )

    conflict_validation_rows = []

    print("\nConflict-group distribution:")

    for split in SPLITS:

        groups = int(
            conflict_split_distribution.loc[
                split,
                "count",
            ]
        )

        records = int(
            conflict_split_distribution.loc[
                split,
                "sum",
            ]
        )

        conflict_validation_rows.append(
            {
                "split": split,
                "conflict_groups": groups,
                "conflict_records": records,
            }
        )

        print(
            f"{split:12s}: "
            f"{groups:,} conflict groups | "
            f"{records:,} conflict records"
        )

    print(
        f"\n✓ Conflict groups: "
        f"{conflict_group_count:,}"
    )

    print(
        f"✓ Conflict records: "
        f"{conflict_record_count:,}"
    )

    # ------------------------------------------------------------------------
    # 7. CROSS-SOURCE GROUPS
    # ------------------------------------------------------------------------

    print(
        "\n[7/10] Validating cross-source-file groups..."
    )

    cross_source_mask = (
        merged["source_file_count"] > 1
    )

    cross_source_groups = merged[
        cross_source_mask
    ].copy()

    cross_source_count = len(
        cross_source_groups
    )

    if cross_source_count != EXPECTED_CROSS_SOURCE_GROUPS:
        fail(
            f"Expected {EXPECTED_CROSS_SOURCE_GROUPS:,} "
            f"cross-source groups, found "
            f"{cross_source_count:,}"
        )

    cross_source_split_distribution = (
        cross_source_groups
        .groupby("split")["group_size"]
        .agg(["count", "sum"])
        .reindex(
            SPLITS,
            fill_value=0,
        )
    )

    source_validation_rows = []

    print("\nCross-source group distribution:")

    for split in SPLITS:

        groups = int(
            cross_source_split_distribution.loc[
                split,
                "count",
            ]
        )

        records = int(
            cross_source_split_distribution.loc[
                split,
                "sum",
            ]
        )

        source_validation_rows.append(
            {
                "split": split,
                "cross_source_groups": groups,
                "cross_source_records": records,
            }
        )

        print(
            f"{split:12s}: "
            f"{groups:,} cross-source groups | "
            f"{records:,} records"
        )

    print(
        f"\n✓ Cross-source groups preserved: "
        f"{cross_source_count:,}"
    )

    # ------------------------------------------------------------------------
    # 8. SPLIT GROUP INTERSECTION
    # ------------------------------------------------------------------------

    print(
        "\n[8/10] Checking split group intersections..."
    )

    train_ids = set(
        merged.loc[
            merged["split"] == "train",
            "feature_group_id",
        ]
    )

    validation_ids = set(
        merged.loc[
            merged["split"] == "validation",
            "feature_group_id",
        ]
    )

    test_ids = set(
        merged.loc[
            merged["split"] == "test",
            "feature_group_id",
        ]
    )

    train_validation_overlap = (
        train_ids
        & validation_ids
    )

    train_test_overlap = (
        train_ids
        & test_ids
    )

    validation_test_overlap = (
        validation_ids
        & test_ids
    )

    if train_validation_overlap:
        fail(
            "Train/validation group intersection "
            "is not empty"
        )

    if train_test_overlap:
        fail(
            "Train/test group intersection "
            "is not empty"
        )

    if validation_test_overlap:
        fail(
            "Validation/test group intersection "
            "is not empty"
        )

    print(
        "✓ Train ∩ Validation = 0"
    )

    print(
        "✓ Train ∩ Test = 0"
    )

    print(
        "✓ Validation ∩ Test = 0"
    )

    # ------------------------------------------------------------------------
    # 9. DETERMINISTIC FINGERPRINT
    # ------------------------------------------------------------------------

    print(
        "\n[9/10] Generating deterministic "
        "assignment fingerprint..."
    )

    assignment_fingerprint = (
        stable_hash_dataframe(
            assignments,
            [
                "feature_group_id",
                "split",
            ],
        )
    )

    print(
        f"Assignment fingerprint: "
        f"{assignment_fingerprint}"
    )

    # ------------------------------------------------------------------------
    # 10. SAVE REPORTS
    # ------------------------------------------------------------------------

    print(
        "\n[10/10] Writing validation reports..."
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_distribution.to_csv(
        OUTPUT_LABEL_DISTRIBUTION,
        index=False,
    )

    pd.DataFrame(
        split_rows
    ).to_csv(
        OUTPUT_GROUP_VALIDATION,
        index=False,
    )

    pd.DataFrame(
        conflict_validation_rows
    ).to_csv(
        OUTPUT_CONFLICT_VALIDATION,
        index=False,
    )

    pd.DataFrame(
        source_validation_rows
    ).to_csv(
        OUTPUT_SOURCE_VALIDATION,
        index=False,
    )

    summary = {
        "task": "13-D.1",
        "status": "PASS",
        "random_state": RANDOM_STATE,

        "expected_groups": EXPECTED_GROUPS,
        "actual_groups": unique_groups,

        "expected_records": EXPECTED_RECORDS,
        "actual_records": split_record_total,

        "metadata_record_total": metadata_record_total,

        "target_ratios": TARGET_RATIOS,

        "split_record_counts": {
            split: int(
                split_record_counts[split]
            )
            for split in SPLITS
        },

        "split_group_counts": {
            split: int(
                split_group_counts[split]
            )
            for split in SPLITS
        },

        "conflict_groups": conflict_group_count,
        "conflict_records": conflict_record_count,

        "cross_source_groups": cross_source_count,

        "expected_labels": len(
            EXPECTED_LABELS
        ),

        "rare_class_threshold": (
            RARE_CLASS_THRESHOLD
        ),

        "assignment_fingerprint": (
            assignment_fingerprint
        ),

        "actual_datasets_materialized": False,

        "validation_checks": {
            "unique_group_assignment": True,
            "no_duplicate_group_ids": True,
            "all_groups_accounted": True,
            "all_records_accounted": True,
            "all_labels_present": True,
            "no_unexpected_labels": True,
            "conflict_groups_preserved": True,
            "cross_source_groups_preserved": True,
            "train_validation_overlap": 0,
            "train_test_overlap": 0,
            "validation_test_overlap": 0,
            "deterministic_fingerprint_generated": True,
        },
    }

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # FINAL STATUS
    # ------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TASK 13-D.1 VALIDATION SUMMARY")
    print("=" * 80)

    print(
        "✓ Every feature group has exactly one split"
    )

    print(
        "✓ No duplicate group assignments"
    )

    print(
        "✓ All expected groups accounted for"
    )

    print(
        "✓ All records accounted for exactly once"
    )

    print(
        "✓ All 15 expected labels present"
    )

    print(
        "✓ No unexpected labels"
    )

    print(
        "✓ Rare classes inspected"
    )

    print(
        "✓ 697 conflict groups preserved"
    )

    print(
        "✓ 6,666 conflict records preserved"
    )

    print(
        "✓ 34,175 cross-source groups preserved"
    )

    print(
        "✓ Train/validation/test group "
        "intersections are zero"
    )

    print(
        "✓ Deterministic assignment fingerprint generated"
    )

    print(
        "✓ Actual datasets NOT materialized"
    )

    print("\nReports:")

    print(
        f"  {OUTPUT_SUMMARY}"
    )

    print(
        f"  {OUTPUT_LABEL_DISTRIBUTION}"
    )

    print(
        f"  {OUTPUT_GROUP_VALIDATION}"
    )

    print(
        f"  {OUTPUT_CONFLICT_VALIDATION}"
    )

    print(
        f"  {OUTPUT_SOURCE_VALIDATION}"
    )

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()