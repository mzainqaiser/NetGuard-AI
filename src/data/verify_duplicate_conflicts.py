"""
NetGuard AI - Task 12-C.1
Direct Verification of Duplicate Feature/Label Conflicts

Purpose:
- Directly verify candidate feature-level label conflicts.
- Avoid relying solely on 64-bit hash fingerprints.
- Identify identical 78-feature rows carrying different labels.
- Report source files involved in each conflict.
- Do NOT modify or delete any dataset records.
"""

from pathlib import Path
from collections import defaultdict

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

LABEL_COLUMN = "Label"
RAW_LABEL_COLUMN = "Raw_Label"

CHUNK_SIZE = 100_000

EXPECTED_FEATURE_COUNT = 78


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NETGUARD AI - TASK 12-C.1")
    print("DIRECT VERIFICATION OF DUPLICATE FEATURE/LABEL CONFLICTS")
    print("=" * 70)

    files = sorted(NORMALIZED_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No normalized CSV files found in {NORMALIZED_DIR}"
        )

    print(f"\nFiles to inspect: {len(files)}")

    # --------------------------------------------------------
    # Step 1:
    # Build candidate feature groups using the existing
    # feature fingerprint approach.
    #
    # Store the actual feature values for candidate groups.
    # --------------------------------------------------------

    candidate_groups = defaultdict(list)

    total_rows = 0

    print("\nPhase 1: Finding candidate feature duplicates...")

    for file_path in files:

        print(f"\nReading: {file_path.name}")

        for chunk_number, df in enumerate(
            pd.read_csv(file_path, chunksize=CHUNK_SIZE),
            start=1,
        ):

            total_rows += len(df)

            if LABEL_COLUMN not in df.columns:
                raise ValueError(
                    f"{file_path.name}: missing {LABEL_COLUMN}"
                )

            if RAW_LABEL_COLUMN not in df.columns:
                raise ValueError(
                    f"{file_path.name}: missing {RAW_LABEL_COLUMN}"
                )

            feature_columns = [
                col
                for col in df.columns
                if col not in {
                    LABEL_COLUMN,
                    RAW_LABEL_COLUMN,
                }
            ]

            if len(feature_columns) != EXPECTED_FEATURE_COUNT:
                raise ValueError(
                    f"{file_path.name}: expected "
                    f"{EXPECTED_FEATURE_COUNT} features, found "
                    f"{len(feature_columns)}"
                )

            # Generate candidate feature fingerprints.
            feature_hashes = pd.util.hash_pandas_object(
                df[feature_columns],
                index=False,
            )

            # Only keep hashes that occur more than once
            # inside the current chunk.
            hash_counts = feature_hashes.value_counts()

            repeated_hashes = set(
                hash_counts[hash_counts > 1].index.tolist()
            )

            if repeated_hashes:

                for hash_value in repeated_hashes:

                    indices = (
                        feature_hashes[
                            feature_hashes == hash_value
                        ]
                        .index
                        .tolist()
                    )

                    for index in indices:

                        row = df.loc[index]

                        feature_tuple = tuple(
                            row[col]
                            for col in feature_columns
                        )

                        candidate_groups[feature_tuple].append(
                            {
                                "label": str(row[LABEL_COLUMN]),
                                "raw_label": str(
                                    row[RAW_LABEL_COLUMN]
                                ),
                                "source_file": file_path.name,
                            }
                        )

            if chunk_number % 5 == 0:
                print(
                    f"  Processed chunk {chunk_number}: "
                    f"{total_rows:,} rows"
                )

    # --------------------------------------------------------
    # Important:
    # The previous approach only collects repeated values
    # inside individual chunks.
    #
    # Therefore, we need a second pass for cross-chunk and
    # cross-file candidate groups.
    # --------------------------------------------------------

    print(
        "\nPhase 1 complete."
    )

    print(
        "Phase 2: Performing direct full-dataset verification..."
    )

    # --------------------------------------------------------
    # Direct verification using feature tuples.
    #
    # For each feature tuple, store labels and source files.
    #
    # To control memory, only retain feature tuples that are
    # actually repeated.
    # --------------------------------------------------------

    first_seen = {}
    repeated_features = defaultdict(list)

    total_rows = 0

    for file_path in files:

        print(f"\nVerifying: {file_path.name}")

        for chunk_number, df in enumerate(
            pd.read_csv(file_path, chunksize=CHUNK_SIZE),
            start=1,
        ):

            total_rows += len(df)

            feature_columns = [
                col
                for col in df.columns
                if col not in {
                    LABEL_COLUMN,
                    RAW_LABEL_COLUMN,
                }
            ]

            # Convert feature rows into tuples.
            #
            # This is intentionally performed only for this
            # verification task.
            for position, (_, row) in enumerate(
                df.iterrows()
            ):

                feature_tuple = tuple(
                    row[col]
                    for col in feature_columns
                )

                label = str(row[LABEL_COLUMN])

                source_record = {
                    "label": label,
                    "raw_label": str(
                        row[RAW_LABEL_COLUMN]
                    ),
                    "source_file": file_path.name,
                }

                if feature_tuple in first_seen:

                    # First occurrence becomes a repeated group.
                    if feature_tuple not in repeated_features:

                        repeated_features[feature_tuple] = [
                            first_seen[feature_tuple]
                        ]

                    repeated_features[
                        feature_tuple
                    ].append(source_record)

                else:

                    first_seen[feature_tuple] = source_record

            if chunk_number % 5 == 0:
                print(
                    f"  Processed chunk {chunk_number}: "
                    f"{total_rows:,} rows"
                )

    # --------------------------------------------------------
    # Verify conflicting groups
    # --------------------------------------------------------

    print(
        "\nPhase 3: Checking labels for repeated feature vectors..."
    )

    conflict_records = []

    conflict_group_id = 0

    for feature_tuple, records in repeated_features.items():

        labels = sorted(
            set(record["label"] for record in records)
        )

        if len(labels) <= 1:
            continue

        conflict_group_id += 1

        source_files = sorted(
            set(
                record["source_file"]
                for record in records
            )
        )

        label_counts = defaultdict(int)

        for record in records:
            label_counts[record["label"]] += 1

        conflict_records.append(
            {
                "conflict_group_id": conflict_group_id,
                "occurrences": len(records),
                "label_count": len(labels),
                "labels": " | ".join(labels),
                "label_counts": " | ".join(
                    f"{label}: {count}"
                    for label, count
                    in sorted(label_counts.items())
                ),
                "source_file_count": len(source_files),
                "source_files": " | ".join(source_files),
            }
        )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_path = (
        REPORTS_DIR
        / "verified_feature_label_conflicts.csv"
    )

    conflict_df = pd.DataFrame(conflict_records)

    conflict_df.to_csv(
        report_path,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    total_conflict_groups = len(conflict_records)

    total_conflict_rows = sum(
        record["occurrences"]
        for record in conflict_records
    )

    cross_file_groups = sum(
        1
        for record in conflict_records
        if record["source_file_count"] > 1
    )

    same_file_groups = (
        total_conflict_groups - cross_file_groups
    )

    print("\n" + "=" * 70)
    print("DIRECT VERIFICATION RESULTS")
    print("=" * 70)

    print(
        f"Total records checked: "
        f"{total_rows:,}"
    )

    print(
        f"Verified conflicting feature groups: "
        f"{total_conflict_groups:,}"
    )

    print(
        f"Rows involved in verified conflicts: "
        f"{total_conflict_rows:,}"
    )

    print(
        f"Conflicting groups across multiple files: "
        f"{cross_file_groups:,}"
    )

    print(
        f"Conflicting groups within one file: "
        f"{same_file_groups:,}"
    )

    print("\nReport created:")
    print(f"  {report_path}")

    print("\n" + "=" * 70)

    if total_conflict_groups == 697:
        print(
            "RESULT: All 697 candidate conflict groups "
            "were directly verified."
        )
    elif total_conflict_groups > 0:
        print(
            "RESULT: Conflicting feature groups were verified."
        )
        print(
            f"Candidate groups from fingerprint analysis: 697"
        )
        print(
            f"Directly verified groups: "
            f"{total_conflict_groups}"
        )
    else:
        print(
            "RESULT: No conflicting feature groups were "
            "directly verified."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()