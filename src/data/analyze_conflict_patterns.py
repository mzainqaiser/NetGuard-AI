"""
NetGuard AI - Task 12-C.2
Verified Feature/Label Conflict Pattern Analysis

Purpose:
- Analyze the 697 verified feature-level label conflicts.
- Quantify labels involved in conflicts.
- Quantify label-pair conflict patterns.
- Analyze same-file vs cross-file conflicts.
- Produce reports for data leakage assessment.
"""

from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_FILE = REPORTS_DIR / "verified_feature_label_conflicts.csv"


# ============================================================
# OUTPUT FILES
# ============================================================

LABEL_INVOLVEMENT_REPORT = (
    REPORTS_DIR / "conflict_label_involvement.csv"
)

LABEL_PAIR_REPORT = (
    REPORTS_DIR / "conflict_label_pairs.csv"
)

FILE_PATTERN_REPORT = (
    REPORTS_DIR / "conflict_file_patterns.csv"
)

SUMMARY_REPORT = (
    REPORTS_DIR / "conflict_pattern_summary.csv"
)


# ============================================================
# HELPERS
# ============================================================

def parse_labels(value):
    """Convert pipe-separated labels into a sorted list."""

    if pd.isna(value):
        return []

    return sorted(
        label.strip()
        for label in str(value).split("|")
        if label.strip()
    )


def parse_label_counts(value):
    """
    Parse:
        BENIGN: 10 | DoS Hulk: 5
    into:
        {"BENIGN": 10, "DoS Hulk": 5}
    """

    result = {}

    if pd.isna(value):
        return result

    for item in str(value).split("|"):

        item = item.strip()

        if ":" not in item:
            continue

        label, count = item.rsplit(":", 1)

        result[label.strip()] = int(count.strip())

    return result


def parse_source_files(value):
    """Convert pipe-separated source files into a list."""

    if pd.isna(value):
        return []

    return sorted(
        file_name.strip()
        for file_name in str(value).split("|")
        if file_name.strip()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NETGUARD AI - TASK 12-C.2")
    print("VERIFIED CONFLICT PATTERN ANALYSIS")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required report not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "conflict_group_id",
        "occurrences",
        "label_count",
        "labels",
        "label_counts",
        "source_file_count",
        "source_files",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    print(
        f"\nVerified conflict groups loaded: {len(df):,}"
    )

    # ========================================================
    # 1. BASIC SUMMARY
    # ========================================================

    total_groups = len(df)

    total_rows = int(
        df["occurrences"].sum()
    )

    cross_file_groups = int(
        (df["source_file_count"] > 1).sum()
    )

    same_file_groups = int(
        (df["source_file_count"] == 1).sum()
    )

    max_labels = int(
        df["label_count"].max()
    )

    print("\nBasic conflict statistics:")
    print(f"  Conflict groups: {total_groups:,}")
    print(f"  Rows involved: {total_rows:,}")
    print(f"  Cross-file groups: {cross_file_groups:,}")
    print(f"  Same-file groups: {same_file_groups:,}")
    print(f"  Maximum labels in one feature group: {max_labels}")

    # ========================================================
    # 2. LABEL INVOLVEMENT
    # ========================================================

    label_groups = Counter()
    label_rows = Counter()

    label_cross_file_groups = Counter()
    label_same_file_groups = Counter()

    for _, row in df.iterrows():

        labels = parse_labels(row["labels"])
        counts = parse_label_counts(row["label_counts"])

        is_cross_file = row["source_file_count"] > 1

        for label in labels:

            label_groups[label] += 1

            label_rows[label] += counts.get(
                label,
                0,
            )

            if is_cross_file:
                label_cross_file_groups[label] += 1
            else:
                label_same_file_groups[label] += 1

    label_records = []

    for label in sorted(label_groups):

        label_records.append(
            {
                "label": label,
                "conflict_groups": label_groups[label],
                "rows_in_conflicts": label_rows[label],
                "cross_file_conflict_groups":
                    label_cross_file_groups[label],
                "same_file_conflict_groups":
                    label_same_file_groups[label],
            }
        )

    label_df = pd.DataFrame(label_records)

    label_df = label_df.sort_values(
        by=[
            "conflict_groups",
            "rows_in_conflicts",
        ],
        ascending=False,
    )

    label_df.to_csv(
        LABEL_INVOLVEMENT_REPORT,
        index=False,
    )

    # ========================================================
    # 3. LABEL PAIR ANALYSIS
    # ========================================================

    pair_groups = Counter()
    pair_rows = Counter()

    pair_cross_file = Counter()
    pair_same_file = Counter()

    for _, row in df.iterrows():

        labels = parse_labels(row["labels"])

        counts = parse_label_counts(
            row["label_counts"]
        )

        for label_a, label_b in combinations(labels, 2):

            pair = (label_a, label_b)

            pair_groups[pair] += 1

            pair_rows[pair] += (
                counts.get(label_a, 0)
                + counts.get(label_b, 0)
            )

            if row["source_file_count"] > 1:
                pair_cross_file[pair] += 1
            else:
                pair_same_file[pair] += 1

    pair_records = []

    for pair in pair_groups:

        label_a, label_b = pair

        pair_records.append(
            {
                "label_a": label_a,
                "label_b": label_b,
                "conflict_groups": pair_groups[pair],
                "rows_in_conflicts": pair_rows[pair],
                "cross_file_groups":
                    pair_cross_file[pair],
                "same_file_groups":
                    pair_same_file[pair],
            }
        )

    pair_df = pd.DataFrame(pair_records)

    if not pair_df.empty:

        pair_df = pair_df.sort_values(
            by=[
                "conflict_groups",
                "rows_in_conflicts",
            ],
            ascending=False,
        )

    pair_df.to_csv(
        LABEL_PAIR_REPORT,
        index=False,
    )

    # ========================================================
    # 4. SOURCE FILE PATTERN ANALYSIS
    # ========================================================

    file_groups = Counter()

    for _, row in df.iterrows():

        source_files = parse_source_files(
            row["source_files"]
        )

        for source_file in source_files:
            file_groups[source_file] += 1

    file_records = [
        {
            "source_file": file_name,
            "conflict_groups": count,
        }
        for file_name, count
        in file_groups.items()
    ]

    file_df = pd.DataFrame(file_records)

    if not file_df.empty:

        file_df = file_df.sort_values(
            "conflict_groups",
            ascending=False,
        )

    file_df.to_csv(
        FILE_PATTERN_REPORT,
        index=False,
    )

    # ========================================================
    # 5. SUMMARY REPORT
    # ========================================================

    summary_records = [
        {
            "metric": "verified_conflict_groups",
            "value": total_groups,
        },
        {
            "metric": "rows_in_conflict_groups",
            "value": total_rows,
        },
        {
            "metric": "cross_file_conflict_groups",
            "value": cross_file_groups,
        },
        {
            "metric": "same_file_conflict_groups",
            "value": same_file_groups,
        },
        {
            "metric": "cross_file_percentage",
            "value": (
                cross_file_groups / total_groups * 100
                if total_groups
                else 0
            ),
        },
        {
            "metric": "same_file_percentage",
            "value": (
                same_file_groups / total_groups * 100
                if total_groups
                else 0
            ),
        },
        {
            "metric": "maximum_labels_per_feature_group",
            "value": max_labels,
        },
        {
            "metric": "unique_labels_involved",
            "value": len(label_groups),
        },
        {
            "metric": "unique_label_pairs",
            "value": len(pair_groups),
        },
    ]

    summary_df = pd.DataFrame(summary_records)

    summary_df.to_csv(
        SUMMARY_REPORT,
        index=False,
    )

    # ========================================================
    # 6. CONSOLE OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("LABEL INVOLVEMENT")
    print("=" * 70)

    print(
        label_df.to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("TOP LABEL-CONFLICT PAIRS")
    print("=" * 70)

    if pair_df.empty:
        print("No label pairs found.")
    else:
        print(
            pair_df.head(20).to_string(
                index=False
            )
        )

    print("\n" + "=" * 70)
    print("SOURCE FILE INVOLVEMENT")
    print("=" * 70)

    print(
        file_df.to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("REPORTS CREATED")
    print("=" * 70)

    print(f"  {LABEL_INVOLVEMENT_REPORT}")
    print(f"  {LABEL_PAIR_REPORT}")
    print(f"  {FILE_PATTERN_REPORT}")
    print(f"  {SUMMARY_REPORT}")

    print("\n" + "=" * 70)
    print("TASK 12-C.2 STATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()