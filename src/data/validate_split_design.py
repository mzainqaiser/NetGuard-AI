"""
NetGuard AI - Task 13-C
Validate Group Metadata & Design Group-Aware Split

This task:
- Validates feature-group metadata.
- Examines class/group distributions.
- Examines multi-label groups.
- Estimates whether a 70/15/15 group-aware split is practical.
- Does NOT create train/validation/test datasets.
"""

from pathlib import Path
import json

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GROUP_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "group_metadata"
    / "feature_group_metadata.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED_RECORDS = 2_830_743
EXPECTED_FEATURE_GROUPS = 2_521_725
EXPECTED_CONFLICT_GROUPS = 697
EXPECTED_CONFLICT_ROWS = 6_666

EXPECTED_LABEL_COUNT = 15

TARGET_TRAIN = 0.70
TARGET_VALIDATION = 0.15
TARGET_TEST = 0.15

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def split_labels(value):
    """Convert pipe-separated label string into a list."""

    if pd.isna(value) or not str(value).strip():
        return []

    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NETGUARD AI - TASK 13-C")
    print("VALIDATE GROUP METADATA & DESIGN SPLIT")
    print("=" * 70)

    # ========================================================
    # LOAD METADATA
    # ========================================================

    if not GROUP_METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Group metadata not found:\n"
            f"{GROUP_METADATA_PATH}"
        )

    print("\nLoading group metadata...")

    df = pd.read_csv(
        GROUP_METADATA_PATH
    )

    print(
        f"Metadata rows loaded: "
        f"{len(df):,}"
    )

    # ========================================================
    # STRUCTURE VALIDATION
    # ========================================================

    required_columns = {
        "feature_group_id",
        "row_count",
        "label_count",
        "labels",
        "raw_labels",
        "source_file_count",
        "source_files",
        "is_label_conflict",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required metadata columns: "
            f"{sorted(missing_columns)}"
        )

    print("\n" + "=" * 70)
    print("1. STRUCTURE VALIDATION")
    print("=" * 70)

    print(
        f"Unique feature groups: "
        f"{len(df):,}"
    )

    print(
        f"Expected feature groups: "
        f"{EXPECTED_FEATURE_GROUPS:,}"
    )

    if len(df) != EXPECTED_FEATURE_GROUPS:

        raise RuntimeError(
            "Feature-group count does not match "
            "Task 13-B result."
        )

    total_records = int(
        df["row_count"].sum()
    )

    print(
        f"Grouped records: "
        f"{total_records:,}"
    )

    print(
        f"Expected records: "
        f"{EXPECTED_RECORDS:,}"
    )

    if total_records != EXPECTED_RECORDS:

        raise RuntimeError(
            "Grouped record count does not match "
            "expected dataset size."
        )

    duplicate_group_ids = int(
        df["feature_group_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate group IDs in metadata: "
        f"{duplicate_group_ids}"
    )

    if duplicate_group_ids != 0:

        raise RuntimeError(
            "Feature group IDs are not unique."
        )

    # ========================================================
    # GROUP SIZE ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("2. GROUP SIZE ANALYSIS")
    print("=" * 70)

    group_sizes = df["row_count"]

    print(
        f"Minimum group size: "
        f"{group_sizes.min():,}"
    )

    print(
        f"Maximum group size: "
        f"{group_sizes.max():,}"
    )

    print(
        f"Mean group size: "
        f"{group_sizes.mean():.3f}"
    )

    print(
        f"Median group size: "
        f"{group_sizes.median():.0f}"
    )

    print(
        f"Groups with 1 record: "
        f"{(group_sizes == 1).sum():,}"
    )

    print(
        f"Groups with >1 record: "
        f"{(group_sizes > 1).sum():,}"
    )

    print(
        f"Groups with >=10 records: "
        f"{(group_sizes >= 10).sum():,}"
    )

    print(
        f"Groups with >=100 records: "
        f"{(group_sizes >= 100).sum():,}"
    )

    # ========================================================
    # MULTI-LABEL GROUP ANALYSIS
    # ========================================================

    print("\n" + "=" * 70)
    print("3. MULTI-LABEL GROUP VALIDATION")
    print("=" * 70)

    conflict_mask = (
        df["is_label_conflict"]
        .astype(bool)
    )

    conflict_groups = int(
        conflict_mask.sum()
    )

    conflict_rows = int(
        df.loc[
            conflict_mask,
            "row_count",
        ].sum()
    )

    print(
        f"Multi-label groups: "
        f"{conflict_groups:,}"
    )

    print(
        f"Expected multi-label groups: "
        f"{EXPECTED_CONFLICT_GROUPS:,}"
    )

    print(
        f"Rows in multi-label groups: "
        f"{conflict_rows:,}"
    )

    print(
        f"Expected conflict rows: "
        f"{EXPECTED_CONFLICT_ROWS:,}"
    )

    if conflict_groups != EXPECTED_CONFLICT_GROUPS:

        raise RuntimeError(
            "Multi-label group count changed."
        )

    if conflict_rows != EXPECTED_CONFLICT_ROWS:

        raise RuntimeError(
            "Multi-label row count changed."
        )

    max_labels = int(
        df["label_count"].max()
    )

    print(
        f"Maximum labels per group: "
        f"{max_labels}"
    )

    if max_labels != 2:

        raise RuntimeError(
            "Unexpected number of labels "
            "inside a feature group."
        )

    # ========================================================
    # CLASS DISTRIBUTION AT RECORD LEVEL
    # ========================================================

    print("\n" + "=" * 70)
    print("4. RECORD-LEVEL CLASS DISTRIBUTION")
    print("=" * 70)

    label_counts = {}

    for _, row in df.iterrows():

        labels = split_labels(
            row["labels"]
        )

        row_count = int(
            row["row_count"]
        )

        if not labels:
            raise RuntimeError(
                "Found a group with no labels."
            )

        # For normal single-label groups,
        # assign the entire group to its label.
        #
        # Multi-label groups are handled separately.
        if len(labels) == 1:

            label = labels[0]

            label_counts[label] = (
                label_counts.get(label, 0)
                + row_count
            )

    label_distribution = (
        pd.DataFrame(
            [
                {
                    "label": label,
                    "record_count": count,
                }
                for label, count
                in label_counts.items()
            ]
        )
        .sort_values(
            "record_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        label_distribution.to_string(
            index=False
        )
    )

    # ========================================================
    # ALL LABELS INCLUDING MULTI-LABEL GROUP INVOLVEMENT
    # ========================================================

    print("\n" + "=" * 70)
    print("5. LABEL GROUP INVOLVEMENT")
    print("=" * 70)

    label_group_stats = {}

    for _, row in df.iterrows():

        labels = split_labels(
            row["labels"]
        )

        for label in labels:

            if label not in label_group_stats:

                label_group_stats[label] = {
                    "group_count": 0,
                    "record_count": 0,
                    "multi_label_group_count": 0,
                }

            label_group_stats[label][
                "group_count"
            ] += 1

            label_group_stats[label][
                "record_count"
            ] += int(
                row["row_count"]
            )

            if len(labels) > 1:

                label_group_stats[label][
                    "multi_label_group_count"
                ] += 1

    label_group_df = pd.DataFrame(
        [
            {
                "label": label,
                "group_count": values[
                    "group_count"
                ],
                "record_count": values[
                    "record_count"
                ],
                "multi_label_group_count": values[
                    "multi_label_group_count"
                ],
            }
            for label, values
            in label_group_stats.items()
        ]
    )

    label_group_df = (
        label_group_df
        .sort_values(
            "record_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        label_group_df.to_string(
            index=False
        )
    )

    # ========================================================
    # TARGET SPLIT CALCULATIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("6. TARGET 70/15/15 SPLIT")
    print("=" * 70)

    targets = {
        "train": TARGET_TRAIN,
        "validation": TARGET_VALIDATION,
        "test": TARGET_TEST,
    }

    split_targets = []

    for split_name, ratio in targets.items():

        target_records = (
            EXPECTED_RECORDS * ratio
        )

        split_targets.append(
            {
                "split": split_name,
                "target_ratio": ratio,
                "target_records": round(
                    target_records
                ),
            }
        )

        print(
            f"{split_name.capitalize():12s}: "
            f"{ratio:.0%} ≈ "
            f"{target_records:,.0f} records"
        )

    split_target_df = pd.DataFrame(
        split_targets
    )

    # ========================================================
    # LARGE GROUP CHECK
    # ========================================================

    print("\n" + "=" * 70)
    print("7. GROUP-SIZE CONSTRAINT CHECK")
    print("=" * 70)

    largest_group = int(
        group_sizes.max()
    )

    print(
        f"Largest feature group: "
        f"{largest_group:,} records"
    )

    print(
        f"Target validation size: "
        f"{EXPECTED_RECORDS * TARGET_VALIDATION:,.0f}"
    )

    print(
        f"Target test size: "
        f"{EXPECTED_RECORDS * TARGET_TEST:,.0f}"
    )

    # ========================================================
    # DESIGN DECISION
    # ========================================================

    print("\n" + "=" * 70)
    print("8. SPLIT DESIGN DECISION")
    print("=" * 70)

    print(
        "Primary grouping unit:"
    )

    print(
        "  feature_group_id"
    )

    print(
        "\nGrouping rule:"
    )

    print(
        "  All records with identical 78-feature "
        "traffic vectors remain in one partition."
    )

    print(
        "\nTarget allocation:"
    )

    print(
        "  Train      ≈ 70%"
    )

    print(
        "  Validation ≈ 15%"
    )

    print(
        "  Test       ≈ 15%"
    )

    print(
        "\nImportant:"
    )

    print(
        "  Ratios are targets, not exact requirements."
    )

    print(
        "  Complete feature groups must remain intact."
    )

    print(
        "  Multi-label groups must remain intact."
    )

    print(
        "  Source-file boundaries will NOT define splits."
    )

    print(
        "  The test partition will remain untouched "
        "until final model evaluation."
    )

    # ========================================================
    # PROPOSED SPLIT ALGORITHM
    # ========================================================

    print("\n" + "=" * 70)
    print("9. PROPOSED ALLOCATION METHOD")
    print("=" * 70)

    print(
        "The actual split will use a stratified, "
        "group-aware allocation."
    )

    print(
        "\nStage 1:"
    )

    print(
        "  Allocate complete feature groups toward "
        "approximately 70% training data."
    )

    print(
        "\nStage 2:"
    )

    print(
        "  Split the remaining approximately 30% "
        "into validation and test targets."
    )

    print(
        "\nStage 3:"
    )

    print(
        "  Check class distribution and rare-class "
        "coverage in all partitions."
    )

    print(
        "\nStage 4:"
    )

    print(
        "  Verify that no feature_group_id occurs "
        "in more than one partition."
    )

    # ========================================================
    # RARE CLASS CHECK
    # ========================================================

    print("\n" + "=" * 70)
    print("10. RARE CLASS CONSIDERATIONS")
    print("=" * 70)

    rare_classes = label_group_df[
        label_group_df["record_count"] <= 1000
    ].copy()

    if len(rare_classes) > 0:

        print(
            "Classes with <= 1,000 records:"
        )

        print(
            rare_classes.to_string(
                index=False
            )
        )

    else:

        print(
            "No classes contain <= 1,000 records."
        )

    print(
        "\nRare classes will be explicitly checked "
        "after the candidate split is generated."
    )

    # ========================================================
    # SAVE REPORTS
    # ========================================================

    label_distribution_path = (
        REPORTS_DIR
        / "split_design_label_distribution.csv"
    )

    label_group_path = (
        REPORTS_DIR
        / "split_design_group_statistics.csv"
    )

    target_path = (
        REPORTS_DIR
        / "split_design_targets.csv"
    )

    summary_path = (
        REPORTS_DIR
        / "split_design_summary.json"
    )

    label_distribution.to_csv(
        label_distribution_path,
        index=False,
    )

    label_group_df.to_csv(
        label_group_path,
        index=False,
    )

    split_target_df.to_csv(
        target_path,
        index=False,
    )

    summary = {
        "status": "PASS",
        "records": EXPECTED_RECORDS,
        "feature_groups": EXPECTED_FEATURE_GROUPS,
        "conflict_groups": EXPECTED_CONFLICT_GROUPS,
        "conflict_rows": EXPECTED_CONFLICT_ROWS,
        "target_train_ratio": TARGET_TRAIN,
        "target_validation_ratio": TARGET_VALIDATION,
        "target_test_ratio": TARGET_TEST,
        "random_state": RANDOM_STATE,
        "grouping_key": "feature_group_id",
        "grouping_features": EXPECTED_FEATURE_GROUPS,
        "largest_group_size": largest_group,
        "split_ratios_are_targets": True,
        "source_file_split": False,
        "multi_label_groups_kept_intact": True,
        "test_reserved_for_final_evaluation": True,
    }

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
    print("TASK 13-C VALIDATION STATUS")
    print("=" * 70)

    print(
        "✓ Feature-group count validated"
    )

    print(
        "✓ Record count validated"
    )

    print(
        "✓ Group IDs are unique"
    )

    print(
        "✓ 697 multi-label groups validated"
    )

    print(
        "✓ 6,666 conflict rows validated"
    )

    print(
        "✓ 15-class structure inspected"
    )

    print(
        "✓ 70/15/15 target defined"
    )

    print(
        "✓ Group-aware split strategy defined"
    )

    print(
        "✓ No actual dataset split created"
    )

    print("\nSTATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()