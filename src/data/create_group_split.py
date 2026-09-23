"""
NetGuard AI - Task 13-D.2
Improved Group-Aware Stratified Split Generator

Purpose
-------
Create a leakage-controlled 70/15/15 train/validation/test assignment
at the feature-group level.

Important design principles
----------------------------
1. Feature groups are never split across partitions.
2. Multi-label/conflict groups are allocated first.
3. Cross-source groups are explicitly balanced.
4. Rare classes are explicitly protected.
5. Single-label groups are then allocated according to remaining
   per-class record targets.
6. Group size is considered during allocation.
7. Random state is fixed for reproducibility.
8. Actual train/validation/test datasets are NOT materialized.

Input
-----
data/group_metadata/feature_group_metadata.csv

Output
------
data/group_metadata/feature_group_split_assignments.parquet
reports/group_split_assignment_summary.json
reports/group_split_record_counts.csv
"""

from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "group_metadata"
    / "feature_group_metadata.csv"
)

OUTPUT_ASSIGNMENT = (
    PROJECT_ROOT
    / "data"
    / "group_metadata"
    / "feature_group_split_assignments.parquet"
)

OUTPUT_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "group_split_assignment_summary.json"
)

OUTPUT_RECORD_COUNTS = (
    PROJECT_ROOT
    / "reports"
    / "group_split_record_counts.csv"
)

RANDOM_STATE = 42

SPLITS = [
    "train",
    "validation",
    "test",
]

TARGET_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

EXPECTED_RECORDS = 2_830_743
EXPECTED_GROUPS = 2_521_725

EXPECTED_CONFLICT_GROUPS = 697
EXPECTED_CONFLICT_ROWS = 6_666

EXPECTED_CROSS_SOURCE_GROUPS = 34_175

RARE_CLASS_THRESHOLD = 1_000

# Weight used when selecting a partition.
#
# Label balance is the primary objective.
# Overall split balance is secondary.
LABEL_WEIGHT = 0.75
GLOBAL_WEIGHT = 0.25

# Additional importance for conflict/cross-source groups.
CONFLICT_WEIGHT = 3.0
CROSS_SOURCE_WEIGHT = 1.5

# Prevent tiny numerical differences from causing unstable decisions.
EPSILON = 1e-12


# ============================================================================
# HELPERS
# ============================================================================

def fail(message: str):
    print("\nSTATUS: FAIL")
    print(f"ERROR: {message}")
    raise SystemExit(1)


def parse_labels(value):
    """
    Convert the metadata labels field into a list.
    """

    if pd.isna(value):
        return []

    return [
        label.strip()
        for label in str(value).split(" | ")
        if label.strip()
    ]


def stable_group_seed(group_id: str) -> int:
    """
    Generate a deterministic integer seed from feature_group_id.

    This prevents dependence on pandas row ordering while keeping
    random tie-breaking reproducible.
    """

    digest = hashlib.blake2b(
        str(group_id).encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="little",
        signed=False,
    )


def choose_best_split(
    group_size,
    labels,
    current_label_records,
    target_label_records,
    current_total_records,
    target_total_records,
    special_bonus,
    group_id,
):
    """
    Select the split that best reduces current allocation deficits.

    The scoring function considers:

    1. Per-label deficit.
    2. Overall record deficit.
    3. Special-group balancing.
    4. Deterministic tie-breaking.

    Higher score = better partition.
    """

    scores = {}

    for split in SPLITS:

        label_score = 0.0

        for label in labels:

            current = current_label_records[
                label
            ][split]

            target = target_label_records[
                label
            ][split]

            # Normalize deficit by target.
            #
            # Positive = under target.
            # Negative = already above target.
            normalized_deficit = (
                target - current
            ) / max(
                target,
                1.0,
            )

            label_score += normalized_deficit

        if labels:
            label_score /= len(labels)

        current_total = current_total_records[
            split
        ]

        target_total = target_total_records[
            split
        ]

        global_deficit = (
            target_total - current_total
        ) / max(
            target_total,
            1.0,
        )

        score = (
            LABEL_WEIGHT * label_score
            + GLOBAL_WEIGHT * global_deficit
        )

        score += special_bonus.get(
            split,
            0.0,
        )

        scores[split] = score

    # Deterministic tie-breaking.
    #
    # We don't use Python's random hash because PYTHONHASHSEED
    # can vary between processes.
    tie_seed = stable_group_seed(group_id)

    ordered_splits = sorted(
        SPLITS,
        key=lambda split: (
            -scores[split],
            (
                tie_seed
                + SPLITS.index(split)
            )
            % 997,
        ),
    )

    return ordered_splits[0]


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("NETGUARD AI - TASK 13-D.2")
    print("IMPROVED GROUP-AWARE STRATIFIED SPLIT GENERATION")
    print("=" * 80)

    np.random.seed(RANDOM_STATE)

    # ------------------------------------------------------------------------
    # 1. LOAD METADATA
    # ------------------------------------------------------------------------

    print("\n[1/8] Loading feature-group metadata...")

    if not METADATA_FILE.exists():
        fail(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    metadata = pd.read_csv(
        METADATA_FILE
    )

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
        - set(metadata.columns)
    )

    if missing_columns:
        fail(
            "Metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(metadata) != EXPECTED_GROUPS:
        fail(
            f"Expected {EXPECTED_GROUPS:,} groups, "
            f"found {len(metadata):,}"
        )

    print(
        f"✓ Groups loaded: "
        f"{len(metadata):,}"
    )

    # ------------------------------------------------------------------------
    # Normalize internal columns
    # ------------------------------------------------------------------------

    metadata = metadata.rename(
        columns={
            "row_count": "group_size",
        }
    )

    metadata["label_list"] = (
        metadata["labels"]
        .apply(parse_labels)
    )

    metadata["is_multi_label"] = (
        metadata["label_count"] > 1
    )

    metadata["is_cross_source"] = (
        metadata["source_file_count"] > 1
    )

    # ------------------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------------------

    total_records = int(
        metadata["group_size"].sum()
    )

    if total_records != EXPECTED_RECORDS:
        fail(
            f"Expected {EXPECTED_RECORDS:,} records, "
            f"found {total_records:,}"
        )

    conflict_groups = metadata[
        metadata["is_label_conflict"]
    ]

    conflict_count = len(
        conflict_groups
    )

    conflict_records = int(
        conflict_groups["group_size"].sum()
    )

    if conflict_count != EXPECTED_CONFLICT_GROUPS:
        fail(
            f"Expected {EXPECTED_CONFLICT_GROUPS:,} "
            f"conflict groups, found "
            f"{conflict_count:,}"
        )

    if conflict_records != EXPECTED_CONFLICT_ROWS:
        fail(
            f"Expected {EXPECTED_CONFLICT_ROWS:,} "
            f"conflict records, found "
            f"{conflict_records:,}"
        )

    cross_source_groups = metadata[
        metadata["is_cross_source"]
    ]

    cross_source_count = len(
        cross_source_groups
    )

    if cross_source_count != EXPECTED_CROSS_SOURCE_GROUPS:
        fail(
            f"Expected {EXPECTED_CROSS_SOURCE_GROUPS:,} "
            f"cross-source groups, found "
            f"{cross_source_count:,}"
        )

    print(
        f"✓ Total records: "
        f"{total_records:,}"
    )

    print(
        f"✓ Conflict groups: "
        f"{conflict_count:,}"
    )

    print(
        f"✓ Conflict records: "
        f"{conflict_records:,}"
    )

    print(
        f"✓ Cross-source groups: "
        f"{cross_source_count:,}"
    )

    # ------------------------------------------------------------------------
    # 2. DETERMINE LABEL TOTALS
    # ------------------------------------------------------------------------

    print("\n[2/8] Calculating per-label record targets...")

    label_totals = {}

    for _, row in metadata.iterrows():

        labels = row["label_list"]
        group_size = int(
            row["group_size"]
        )

        for label in labels:

            label_totals[label] = (
                label_totals.get(
                    label,
                    0,
                )
                + group_size
            )

    labels = sorted(
        label_totals.keys()
    )

    print(
        f"✓ Labels detected: "
        f"{len(labels)}"
    )

    # Target records per label and split.
    target_label_records = {}

    for label in labels:

        total = label_totals[label]

        target_label_records[label] = {}

        for split in SPLITS:

            target_label_records[label][split] = (
                total
                * TARGET_RATIOS[split]
            )

    # Global target.
    target_total_records = {
        split:
            total_records
            * TARGET_RATIOS[split]
        for split in SPLITS
    }

    # ------------------------------------------------------------------------
    # 3. INITIALIZE TRACKERS
    # ------------------------------------------------------------------------

    print("\n[3/8] Initializing allocation trackers...")

    current_label_records = {
        label: {
            split: 0
            for split in SPLITS
        }
        for label in labels
    }

    current_total_records = {
        split: 0
        for split in SPLITS
    }

    assignment_rows = []

    print("✓ Allocation trackers initialized")

    # ------------------------------------------------------------------------
    # 4. ALLOCATE MULTI-LABEL / CONFLICT GROUPS FIRST
    # ------------------------------------------------------------------------

    print(
        "\n[4/8] Allocating multi-label/conflict groups..."
    )

    multi_label_groups = metadata[
        metadata["is_multi_label"]
    ].copy()

    # Large groups first.
    #
    # This is important because a large indivisible group can significantly
    # change a partition's class distribution.
    multi_label_groups = (
        multi_label_groups
        .sort_values(
            by=[
                "group_size",
                "feature_group_id",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
    )

    conflict_assignment_counts = {
        split: 0
        for split in SPLITS
    }

    conflict_assignment_records = {
        split: 0
        for split in SPLITS
    }

    for _, row in multi_label_groups.iterrows():

        group_id = row[
            "feature_group_id"
        ]

        group_size = int(
            row["group_size"]
        )

        group_labels = row[
            "label_list"
        ]

        # Strong balancing bonus for conflict groups.
        #
        # The objective is to prevent the previous situation where
        # almost all 697 conflict groups went to validation/test.
        special_bonus = {}

        for split in SPLITS:

            assigned = conflict_assignment_records[
                split
            ]

            target = (
                conflict_records
                * TARGET_RATIOS[split]
            )

            deficit = (
                target - assigned
            ) / max(
                target,
                1.0,
            )

            special_bonus[split] = (
                CONFLICT_WEIGHT
                * deficit
            )

        split = choose_best_split(
            group_size=group_size,
            labels=group_labels,
            current_label_records=current_label_records,
            target_label_records=target_label_records,
            current_total_records=current_total_records,
            target_total_records=target_total_records,
            special_bonus=special_bonus,
            group_id=group_id,
        )

        assignment_rows.append(
            {
                "feature_group_id": group_id,
                "split": split,
            }
        )

        current_total_records[
            split
        ] += group_size

        conflict_assignment_counts[
            split
        ] += 1

        conflict_assignment_records[
            split
        ] += group_size

        for label in group_labels:

            current_label_records[
                label
            ][split] += group_size

    print(
        "\nConflict-group allocation:"
    )

    for split in SPLITS:

        print(
            f"  {split:12s}: "
            f"{conflict_assignment_counts[split]:>5,} groups | "
            f"{conflict_assignment_records[split]:>6,} records"
        )

    # ------------------------------------------------------------------------
    # 5. ALLOCATE CROSS-SOURCE SINGLE-LABEL GROUPS
    # ------------------------------------------------------------------------

    print(
        "\n[5/8] Allocating cross-source single-label groups..."
    )

    cross_source_single = metadata[
        metadata["is_cross_source"]
        & (~metadata["is_multi_label"])
    ].copy()

    cross_source_single = (
        cross_source_single
        .sort_values(
            by=[
                "group_size",
                "feature_group_id",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
    )

    cross_source_assignment_counts = {
        split: 0
        for split in SPLITS
    }

    cross_source_assignment_records = {
        split: 0
        for split in SPLITS
    }

    for _, row in cross_source_single.iterrows():

        group_id = row[
            "feature_group_id"
        ]

        group_size = int(
            row["group_size"]
        )

        group_labels = row[
            "label_list"
        ]

        special_bonus = {}

        for split in SPLITS:

            assigned = (
                cross_source_assignment_records[
                    split
                ]
            )

            target = (
                (cross_source_count)
                * TARGET_RATIOS[split]
            )

            deficit = (
                target - assigned
            ) / max(
                target,
                1.0,
            )

            special_bonus[split] = (
                CROSS_SOURCE_WEIGHT
                * deficit
            )

        split = choose_best_split(
            group_size=group_size,
            labels=group_labels,
            current_label_records=current_label_records,
            target_label_records=target_label_records,
            current_total_records=current_total_records,
            target_total_records=target_total_records,
            special_bonus=special_bonus,
            group_id=group_id,
        )

        assignment_rows.append(
            {
                "feature_group_id": group_id,
                "split": split,
            }
        )

        current_total_records[
            split
        ] += group_size

        cross_source_assignment_counts[
            split
        ] += 1

        cross_source_assignment_records[
            split
        ] += group_size

        for label in group_labels:

            current_label_records[
                label
            ][split] += group_size

    print(
        "\nCross-source single-label allocation:"
    )

    for split in SPLITS:

        print(
            f"  {split:12s}: "
            f"{cross_source_assignment_counts[split]:>6,} groups | "
            f"{cross_source_assignment_records[split]:>7,} records"
        )

    # ------------------------------------------------------------------------
    # 6. ALLOCATE REMAINING SINGLE-LABEL GROUPS
    # ------------------------------------------------------------------------

    print(
        "\n[6/8] Allocating remaining single-label groups..."
    )

    allocated_ids = {
        row["feature_group_id"]
        for row in assignment_rows
    }

    remaining = metadata[
        ~metadata["feature_group_id"].isin(
            allocated_ids
        )
    ].copy()

    # All remaining groups should be single-label.
    invalid_remaining = remaining[
        remaining["is_multi_label"]
    ]

    if not invalid_remaining.empty:
        fail(
            "Some multi-label groups were not allocated "
            "during the first stage: "
            f"{len(invalid_remaining):,}"
        )

    # ------------------------------------------------------------------------
    # Allocate rare classes first.
    # ------------------------------------------------------------------------

    remaining_label_counts = (
        remaining["label_list"]
        .apply(
            lambda x: x[0]
            if len(x) == 1
            else None
        )
        .value_counts()
    )

    rare_labels = [
        label
        for label in labels
        if label_totals[label]
        <= RARE_CLASS_THRESHOLD
    ]

    rare_remaining = remaining[
        remaining["label_list"].apply(
            lambda x:
                len(x) == 1
                and x[0] in rare_labels
        )
    ].copy()

    normal_remaining = remaining[
        ~remaining["feature_group_id"].isin(
            set(
                rare_remaining[
                    "feature_group_id"
                ]
            )
        )
    ].copy()

    print(
        f"✓ Remaining groups: "
        f"{len(remaining):,}"
    )

    print(
        f"✓ Rare-class groups: "
        f"{len(rare_remaining):,}"
    )

    # ------------------------------------------------------------------------
    # Helper for single-label allocation.
    # ------------------------------------------------------------------------

    def allocate_single_label_groups(
        frame: pd.DataFrame,
        stage_name: str,
    ):

        if frame.empty:
            return

        # Process labels with fewer records first.
        #
        # This protects rare classes from being affected by the much larger
        # BENIGN class.
        label_order = sorted(
            frame["label_list"].apply(
                lambda x: x[0]
            ).unique(),
            key=lambda label: (
                label_totals[label],
                label,
            ),
        )

        for label in label_order:

            label_frame = frame[
                frame["label_list"].apply(
                    lambda x:
                        len(x) == 1
                        and x[0] == label
                )
            ].copy()

            # Largest groups first.
            label_frame = (
                label_frame
                .sort_values(
                    by=[
                        "group_size",
                        "feature_group_id",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                    kind="mergesort",
                )
            )

            for _, row in label_frame.iterrows():

                group_id = row[
                    "feature_group_id"
                ]

                group_size = int(
                    row["group_size"]
                )

                # Current deficit for this label.
                label_deficits = {}

                for split in SPLITS:

                    target = target_label_records[
                        label
                    ][split]

                    current = current_label_records[
                        label
                    ][split]

                    label_deficits[split] = (
                        target - current
                    ) / max(
                        target,
                        1.0,
                    )

                # Overall deficit.
                global_deficits = {}

                for split in SPLITS:

                    target = target_total_records[
                        split
                    ]

                    current = current_total_records[
                        split
                    ]

                    global_deficits[split] = (
                        target - current
                    ) / max(
                        target,
                        1.0,
                    )

                # Strong class-level objective.
                scores = {}

                for split in SPLITS:

                    scores[split] = (
                        LABEL_WEIGHT
                        * label_deficits[split]
                        + GLOBAL_WEIGHT
                        * global_deficits[split]
                    )

                # Deterministic tie-breaking.
                seed = stable_group_seed(
                    group_id
                )

                split = min(
                    SPLITS,
                    key=lambda candidate: (
                        -scores[candidate],
                        (
                            seed
                            + SPLITS.index(
                                candidate
                            )
                        ) % 997,
                    ),
                )

                assignment_rows.append(
                    {
                        "feature_group_id": group_id,
                        "split": split,
                    }
                )

                current_total_records[
                    split
                ] += group_size

                current_label_records[
                    label
                ][split] += group_size

        print(
            f"✓ {stage_name} allocation complete"
        )

    # ------------------------------------------------------------------------
    # Rare labels first.
    # ------------------------------------------------------------------------

    allocate_single_label_groups(
        rare_remaining,
        "Rare-class",
    )

    allocated_ids = {
        row["feature_group_id"]
        for row in assignment_rows
    }

    normal_remaining = normal_remaining[
        ~normal_remaining["feature_group_id"].isin(
            allocated_ids
        )
    ]

    # ------------------------------------------------------------------------
    # Normal labels.
    # ------------------------------------------------------------------------

    allocate_single_label_groups(
        normal_remaining,
        "Normal-class",
    )

    # ------------------------------------------------------------------------
    # 7. BUILD ASSIGNMENT DATAFRAME
    # ------------------------------------------------------------------------

    print(
        "\n[7/8] Building and validating assignment..."
    )

    assignments = pd.DataFrame(
        assignment_rows
    )

    if len(assignments) != EXPECTED_GROUPS:
        fail(
            f"Expected {EXPECTED_GROUPS:,} assignment rows, "
            f"found {len(assignments):,}"
        )

    duplicate_ids = (
        assignments["feature_group_id"]
        .duplicated()
        .sum()
    )

    if duplicate_ids != 0:
        fail(
            f"Duplicate group assignments found: "
            f"{duplicate_ids:,}"
        )

    assignment_ids = set(
        assignments["feature_group_id"]
    )

    metadata_ids = set(
        metadata["feature_group_id"]
    )

    if assignment_ids != metadata_ids:
        fail(
            "Assignment group IDs do not exactly match "
            "metadata group IDs"
        )

    # Merge for final calculations.
    final = metadata.merge(
        assignments,
        on="feature_group_id",
        how="left",
        validate="one_to_one",
    )

    if final["split"].isna().any():
        fail(
            "Some groups have no split assignment"
        )

    invalid_splits = sorted(
        set(final["split"].unique())
        - set(SPLITS)
    )

    if invalid_splits:
        fail(
            f"Invalid split values found: "
            f"{invalid_splits}"
        )

    # ------------------------------------------------------------------------
    # Record counts.
    # ------------------------------------------------------------------------

    split_record_counts = (
        final.groupby("split")["group_size"]
        .sum()
        .reindex(
            SPLITS,
            fill_value=0,
        )
    )

    split_group_counts = (
        final.groupby("split")["feature_group_id"]
        .nunique()
        .reindex(
            SPLITS,
            fill_value=0,
        )
    )

    print("\nFinal record allocation:")

    record_rows = []

    for split in SPLITS:

        records = int(
            split_record_counts[split]
        )

        groups = int(
            split_group_counts[split]
        )

        ratio = (
            records
            / EXPECTED_RECORDS
        )

        target = TARGET_RATIOS[
            split
        ]

        difference = (
            ratio - target
        )

        record_rows.append(
            {
                "split": split,
                "groups": groups,
                "records": records,
                "actual_ratio": ratio,
                "target_ratio": target,
                "difference": difference,
            }
        )

        print(
            f"  {split:12s}: "
            f"{records:>10,} records "
            f"({ratio * 100:8.4f}%) "
            f"target={target * 100:.2f}% "
            f"diff={difference * 100:+.4f}%"
        )

    if int(
        split_record_counts.sum()
    ) != EXPECTED_RECORDS:
        fail(
            "Final split record counts do not "
            "sum to the expected dataset size"
        )

    # ------------------------------------------------------------------------
    # Conflict distribution.
    # ------------------------------------------------------------------------

    final_conflicts = final[
        final["is_label_conflict"]
    ]

    conflict_split_counts = (
        final_conflicts
        .groupby("split")["group_size"]
        .agg(
            ["count", "sum"]
        )
        .reindex(
            SPLITS,
            fill_value=0,
        )
    )

    print(
        "\nFinal conflict-group allocation:"
    )

    for split in SPLITS:

        print(
            f"  {split:12s}: "
            f"{int(conflict_split_counts.loc[split, 'count']):>5,} groups | "
            f"{int(conflict_split_counts.loc[split, 'sum']):>6,} records"
        )

    # ------------------------------------------------------------------------
    # Cross-source distribution.
    # ------------------------------------------------------------------------

    final_cross_source = final[
        final["is_cross_source"]
    ]

    cross_source_split_counts = (
        final_cross_source
        .groupby("split")["group_size"]
        .agg(
            ["count", "sum"]
        )
        .reindex(
            SPLITS,
            fill_value=0,
        )
    )

    print(
        "\nFinal cross-source allocation:"
    )

    for split in SPLITS:

        print(
            f"  {split:12s}: "
            f"{int(cross_source_split_counts.loc[split, 'count']):>6,} groups | "
            f"{int(cross_source_split_counts.loc[split, 'sum']):>7,} records"
        )

    # ------------------------------------------------------------------------
    # Label distribution.
    # ------------------------------------------------------------------------

    print(
        "\nFinal label distribution:"
    )

    label_summary = []

    for label in labels:

        label_mask = final["label_list"].apply(
            lambda values:
                label in values
        )

        label_frame = final[
            label_mask
        ]

        total = int(
            label_frame["group_size"].sum()
        )

        row = {
            "label": label,
            "total_records": total,
        }

        print(
            f"\n{label}"
        )

        for split in SPLITS:

            count = int(
                label_frame[
                    label_frame["split"] == split
                ]["group_size"].sum()
            )

            percentage = (
                count / total
                if total
                else 0.0
            )

            row[
                f"{split}_records"
            ] = count

            row[
                f"{split}_percentage"
            ] = percentage

            print(
                f"  {split:12s}: "
                f"{count:>10,} "
                f"({percentage * 100:6.2f}%)"
            )

        label_summary.append(row)

    # ------------------------------------------------------------------------
    # 8. SAVE OUTPUT
    # ------------------------------------------------------------------------

    print(
        "\n[8/8] Saving improved split assignment..."
    )

    OUTPUT_ASSIGNMENT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignments.to_parquet(
        OUTPUT_ASSIGNMENT,
        index=False,
    )

    pd.DataFrame(
        record_rows
    ).to_csv(
        OUTPUT_RECORD_COUNTS,
        index=False,
    )

    assignment_hash = hashlib.sha256(
        assignments.sort_values(
            [
                "feature_group_id",
                "split",
            ],
            kind="mergesort",
        )
        .to_csv(
            index=False,
            lineterminator="\n",
        )
        .encode("utf-8")
    ).hexdigest()

    summary = {
        "task": "13-D.2",
        "status": "PASS",
        "random_state": RANDOM_STATE,

        "strategy": (
            "multi-label/conflict-first, "
            "cross-source-aware, "
            "rare-class-first, "
            "per-label deficit allocation"
        ),

        "expected_groups": EXPECTED_GROUPS,
        "assigned_groups": len(assignments),

        "expected_records": EXPECTED_RECORDS,
        "assigned_records": int(
            split_record_counts.sum()
        ),

        "target_ratios": TARGET_RATIOS,

        "split_group_counts": {
            split: int(
                split_group_counts[split]
            )
            for split in SPLITS
        },

        "split_record_counts": {
            split: int(
                split_record_counts[split]
            )
            for split in SPLITS
        },

        "conflict_groups": int(
            len(final_conflicts)
        ),

        "conflict_records": int(
            final_conflicts["group_size"].sum()
        ),

        "cross_source_groups": int(
            len(final_cross_source)
        ),

        "cross_source_records": int(
            final_cross_source["group_size"].sum()
        ),

        "assignment_fingerprint": assignment_hash,

        "actual_datasets_materialized": False,

        "notes": [
            "Feature groups remain intact.",
            "Multi-label conflict groups are allocated first.",
            "Cross-source groups are explicitly considered.",
            "Rare classes are allocated before normal classes.",
            "Assignment is deterministic with RANDOM_STATE=42.",
            "Actual train/validation/test datasets are not materialized.",
            "Strict validation must be rerun in Task 13-D.1.",
        ],
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

    print(
        "\n✓ Assignment saved:"
    )

    print(
        f"  {OUTPUT_ASSIGNMENT}"
    )

    print(
        "\n✓ Summary saved:"
    )

    print(
        f"  {OUTPUT_SUMMARY}"
    )

    print(
        "\n✓ Record-count report saved:"
    )

    print(
        f"  {OUTPUT_RECORD_COUNTS}"
    )

    print(
        "\nTASK 13-D.2 STATUS: PASS"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Run Task 13-D.1 strict validation before "
        "accepting this assignment."
    )


if __name__ == "__main__":
    main()