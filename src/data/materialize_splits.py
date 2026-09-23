"""
NetGuard AI — Materialize Validated Group-Aware Splits

Creates:
    data/splits/train/train.parquet
    data/splits/validation/validation.parquet
    data/splits/test/test.parquet

The validated feature-group assignments from Phase 3 are used
to materialize the final train/validation/test datasets.

Output contains:
    - 78 traffic features
    - Label
    - Raw_Label
    - Source_File

feature_group_id is used internally for assignment and is NOT
included in the final ML dataset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NORMALIZED_DIR = (
    PROJECT_ROOT / "data" / "normalized"
)

ASSIGNMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "group_metadata"
    / "feature_group_split_assignments.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "splits"
)

REPORT_DIR = (
    PROJECT_ROOT / "reports"
)

CHUNK_SIZE = 100_000

HASH_DIGEST_SIZE = 16

RANDOM_STATE = 42

LABEL_COLUMN = "Label"
RAW_LABEL_COLUMN = "Raw_Label"
SOURCE_COLUMN = "Source_File"

EXPECTED_FEATURE_COUNT = 78
EXPECTED_SOURCE_COLUMN_COUNT = 80
EXPECTED_OUTPUT_COLUMN_COUNT = 81

EXPECTED_GROUP_COUNT = 2_521_725
EXPECTED_RECORD_COUNT = 2_830_743

EXPECTED_SPLITS = (
    "train",
    "validation",
    "test",
)

# Safety setting.
# Keep False so existing materialized datasets are never
# accidentally overwritten.
OVERWRITE_EXISTING = False


# ============================================================
# Feature-group fingerprint
# ============================================================

def get_feature_columns(
    columns: list[str],
) -> list[str]:
    """
    Return the 78 traffic feature columns.

    Label and Raw_Label are excluded.
    Source_File is also excluded if present.
    """

    excluded = {
        LABEL_COLUMN,
        RAW_LABEL_COLUMN,
        SOURCE_COLUMN,
    }

    feature_columns = [
        column
        for column in columns
        if column not in excluded
    ]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Unexpected feature count.\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Found: {len(feature_columns)}"
        )

    return feature_columns


def normalize_value(value) -> str:
    """
    Convert a feature value to the exact representation
    used by build_feature_groups.py.
    """

    if pd.isna(value):
        return "<NA>"

    if isinstance(value, float):
        return format(value, ".17g")

    return str(value)


def create_feature_group_id(
    row: tuple,
    feature_indices: list[int],
) -> str:
    """
    Create the deterministic BLAKE2b feature-group ID.

    This matches the fingerprint algorithm used by
    src/data/build_feature_groups.py.
    """

    hasher = hashlib.blake2b(
        digest_size=HASH_DIGEST_SIZE
    )

    for index in feature_indices:
        value = normalize_value(
            row[index]
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
# Directory and output handling
# ============================================================

def ensure_output_directories() -> None:
    """Create required directories."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in EXPECTED_SPLITS:
        (
            OUTPUT_DIR / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


def get_output_paths() -> dict[str, Path]:
    """Return final Parquet output paths."""

    return {
        "train": (
            OUTPUT_DIR
            / "train"
            / "train.parquet"
        ),
        "validation": (
            OUTPUT_DIR
            / "validation"
            / "validation.parquet"
        ),
        "test": (
            OUTPUT_DIR
            / "test"
            / "test.parquet"
        ),
    }


def check_existing_outputs(
    output_paths: dict[str, Path],
) -> None:
    """
    Prevent accidental overwriting of existing
    materialized datasets.
    """

    existing = [
        path
        for path in output_paths.values()
        if path.exists()
    ]

    if not existing:
        return

    if not OVERWRITE_EXISTING:
        print(
            "Existing materialized outputs detected:"
        )

        for path in existing:
            print(f"  {path}")

        raise FileExistsError(
            "Materialized datasets already exist. "
            "No files were overwritten."
        )

    print(
        "WARNING: OVERWRITE_EXISTING=True"
    )

    for path in existing:
        path.unlink()


# ============================================================
# Assignment loading
# ============================================================

def load_split_assignments() -> dict[str, str]:
    """
    Load validated feature-group assignments.

    Returns:
        {
            feature_group_id: split
        }
    """

    if not ASSIGNMENT_FILE.exists():
        raise FileNotFoundError(
            "Validated split assignment file not found:\n"
            f"{ASSIGNMENT_FILE}"
        )

    print(
        "Loading validated split assignments..."
    )

    assignment = pd.read_parquet(
        ASSIGNMENT_FILE
    )

    expected_columns = {
        "feature_group_id",
        "split",
    }

    if set(assignment.columns) != expected_columns:
        raise ValueError(
            "Unexpected assignment columns.\n"
            f"Expected: {sorted(expected_columns)}\n"
            f"Found: {sorted(assignment.columns)}"
        )

    if len(assignment) != EXPECTED_GROUP_COUNT:
        raise ValueError(
            "Unexpected feature-group count.\n"
            f"Expected: {EXPECTED_GROUP_COUNT:,}\n"
            f"Found: {len(assignment):,}"
        )

    if (
        assignment["feature_group_id"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate feature_group_id values "
            "found in assignments."
        )

    if assignment["split"].isna().any():
        raise ValueError(
            "Some feature groups have no split assignment."
        )

    invalid_splits = (
        set(
            assignment["split"]
            .unique()
        )
        - set(EXPECTED_SPLITS)
    )

    if invalid_splits:
        raise ValueError(
            "Unexpected split values found:\n"
            f"{sorted(invalid_splits)}"
        )

    lookup = dict(
        zip(
            assignment["feature_group_id"],
            assignment["split"],
        )
    )

    print(
        f"  Feature groups loaded: "
        f"{len(lookup):,}"
    )

    return lookup


# ============================================================
# Schema validation
# ============================================================

def validate_output_schema(
    parquet_path: Path,
) -> None:
    """
    Validate final Parquet schema.
    """

    parquet_file = pq.ParquetFile(
        parquet_path
    )

    columns = (
        parquet_file.schema_arrow.names
    )

    if len(columns) != EXPECTED_OUTPUT_COLUMN_COUNT:
        raise ValueError(
            f"Unexpected column count in "
            f"{parquet_path.name}.\n"
            f"Expected: {EXPECTED_OUTPUT_COLUMN_COUNT}\n"
            f"Found: {len(columns)}"
        )

    required_columns = {
        LABEL_COLUMN,
        RAW_LABEL_COLUMN,
        SOURCE_COLUMN,
    }

    missing = (
        required_columns
        - set(columns)
    )

    if missing:
        raise ValueError(
            "Required columns missing from "
            f"{parquet_path.name}: "
            f"{sorted(missing)}"
        )

    if "feature_group_id" in columns:
        raise ValueError(
            "feature_group_id must not be present "
            "in the final ML dataset."
        )


# ============================================================
# Main materialization
# ============================================================

def materialize() -> None:
    """
    Materialize train, validation, and test datasets.
    """

    print("=" * 70)
    print(
        "NETGUARD AI — DATASET MATERIALIZATION"
    )
    print("=" * 70)
    print()

    ensure_output_directories()

    output_paths = (
        get_output_paths()
    )

    check_existing_outputs(
        output_paths
    )

    normalized_files = sorted(
        NORMALIZED_DIR.glob("*.csv")
    )

    if len(normalized_files) != 8:
        raise ValueError(
            "Expected exactly 8 normalized CSV files.\n"
            f"Found: {len(normalized_files)}"
        )

    print(
        f"Normalized source files: "
        f"{len(normalized_files)}"
    )

    print(
        f"Chunk size: "
        f"{CHUNK_SIZE:,} rows"
    )

    print(
        f"Expected records: "
        f"{EXPECTED_RECORD_COUNT:,}"
    )

    print(
        f"Expected feature groups: "
        f"{EXPECTED_GROUP_COUNT:,}"
    )

    print()

    # --------------------------------------------------------
    # Load validated assignments
    # --------------------------------------------------------

    assignment_lookup = (
        load_split_assignments()
    )

    print()

    # --------------------------------------------------------
    # Initialize Parquet writers
    # --------------------------------------------------------

    writers: dict[
        str,
        pq.ParquetWriter | None,
    ] = {
        split: None
        for split in EXPECTED_SPLITS
    }

    output_counts = {
        split: 0
        for split in EXPECTED_SPLITS
    }

    source_record_counts = {
        source.name: 0
        for source in normalized_files
    }

    source_split_counts = {
        source.name: {
            split: 0
            for split in EXPECTED_SPLITS
        }
        for source in normalized_files
    }

    total_rows_processed = 0

    # --------------------------------------------------------
    # Process normalized files
    # --------------------------------------------------------

    try:

        for file_number, source_file in enumerate(
            normalized_files,
            start=1,
        ):

            print("-" * 70)

            print(
                f"[{file_number}/"
                f"{len(normalized_files)}] "
                f"{source_file.name}"
            )

            print("-" * 70)

            file_rows = 0
            file_chunks = 0

            for chunk_number, chunk in enumerate(
                pd.read_csv(
                    source_file,
                    chunksize=CHUNK_SIZE,
                    low_memory=False,
                ),
                start=1,
            ):

                file_chunks += 1

                print(
                    f"  Chunk {chunk_number}: "
                    f"{len(chunk):,} rows"
                )

                # --------------------------------------------
                # Validate source schema
                # --------------------------------------------

                if (
                    len(chunk.columns)
                    != EXPECTED_SOURCE_COLUMN_COUNT
                ):
                    raise ValueError(
                        "Unexpected source column count "
                        f"in {source_file.name}.\n"
                        f"Expected: "
                        f"{EXPECTED_SOURCE_COLUMN_COUNT}\n"
                        f"Found: {len(chunk.columns)}"
                    )

                required_source_columns = {
                    LABEL_COLUMN,
                    RAW_LABEL_COLUMN,
                }

                missing = (
                    required_source_columns
                    - set(chunk.columns)
                )

                if missing:
                    raise ValueError(
                        "Required columns missing from "
                        f"{source_file.name}: "
                        f"{sorted(missing)}"
                    )

                # --------------------------------------------
                # Identify feature columns
                # --------------------------------------------

                feature_columns = (
                    get_feature_columns(
                        chunk.columns.tolist()
                    )
                )

                feature_indices = [
                    chunk.columns.get_loc(
                        column
                    )
                    for column in feature_columns
                ]

                # --------------------------------------------
                # Recreate feature-group IDs
                # --------------------------------------------

                feature_group_ids = []

                for row in chunk.itertuples(
                    index=False,
                    name=None,
                ):

                    group_id = (
                        create_feature_group_id(
                            row,
                            feature_indices,
                        )
                    )

                    feature_group_ids.append(
                        group_id
                    )

                # --------------------------------------------
                # Look up validated split assignment
                # --------------------------------------------

                splits = [
                    assignment_lookup.get(
                        group_id
                    )
                    for group_id
                    in feature_group_ids
                ]

                missing_assignments = sum(
                    split is None
                    for split in splits
                )

                if missing_assignments:
                    raise ValueError(
                        f"{missing_assignments:,} rows "
                        "could not be matched to a "
                        "validated feature-group assignment "
                        f"in {source_file.name}."
                    )

                # --------------------------------------------
                # Add provenance column
                # --------------------------------------------

                chunk[
                    SOURCE_COLUMN
                ] = source_file.name

                # --------------------------------------------
                # Output column order
                # --------------------------------------------

                output_columns = (
                    chunk.columns.tolist()
                )

                if (
                    "feature_group_id"
                    in output_columns
                ):
                    raise ValueError(
                        "feature_group_id unexpectedly "
                        "exists in source data."
                    )

                if len(output_columns) != (
                    EXPECTED_OUTPUT_COLUMN_COUNT
                ):
                    raise ValueError(
                        "Unexpected output column count "
                        f"after adding {SOURCE_COLUMN}."
                    )

                # --------------------------------------------
                # Write each split
                # --------------------------------------------

                for split in EXPECTED_SPLITS:

                    split_indices = [
                        index
                        for index, value
                        in enumerate(splits)
                        if value == split
                    ]

                    if not split_indices:
                        continue

                    split_chunk = (
                        chunk.iloc[
                            split_indices
                        ][
                            output_columns
                        ].copy()
                    )

                    table = (
                        pa.Table.from_pandas(
                            split_chunk,
                            preserve_index=False,
                        )
                    )

                    # Initialize writer on first chunk.
                    if writers[split] is None:

                        writers[split] = (
                            pq.ParquetWriter(
                                output_paths[split],
                                table.schema,
                                compression="snappy",
                            )
                        )

                    writers[split].write_table(
                        table
                    )

                    row_count = (
                        len(split_chunk)
                    )

                    output_counts[
                        split
                    ] += row_count

                    source_split_counts[
                        source_file.name
                    ][split] += row_count

                # --------------------------------------------
                # Counters
                # --------------------------------------------

                rows_in_chunk = len(chunk)

                file_rows += rows_in_chunk

                total_rows_processed += (
                    rows_in_chunk
                )

                source_record_counts[
                    source_file.name
                ] += rows_in_chunk

            print(
                f"  Completed: "
                f"{file_rows:,} rows "
                f"across {file_chunks} chunks."
            )

            print()

    finally:

        # ----------------------------------------------------
        # Always close Parquet writers
        # ----------------------------------------------------

        for split in EXPECTED_SPLITS:

            writer = writers[split]

            if writer is not None:
                writer.close()

    # --------------------------------------------------------
    # Validate processed record count
    # --------------------------------------------------------

    if (
        total_rows_processed
        != EXPECTED_RECORD_COUNT
    ):
        raise ValueError(
            "Unexpected processed record count.\n"
            f"Expected: {EXPECTED_RECORD_COUNT:,}\n"
            f"Found: {total_rows_processed:,}"
        )

    # --------------------------------------------------------
    # Validate output files
    # --------------------------------------------------------

    split_validation = {}

    for split in EXPECTED_SPLITS:

        path = output_paths[split]

        if not path.exists():
            raise FileNotFoundError(
                "Expected output file was not created:\n"
                f"{path}"
            )

        validate_output_schema(
            path
        )

        file_size_bytes = (
            path.stat().st_size
        )

        split_validation[split] = {
            "path": str(
                path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "records": output_counts[
                split
            ],
            "file_size_bytes": (
                file_size_bytes
            ),
            "file_size_mb": round(
                file_size_bytes
                / (1024 * 1024),
                2,
            ),
        }

    # --------------------------------------------------------
    # Validate total materialized records
    # --------------------------------------------------------

    materialized_total = sum(
        output_counts.values()
    )

    if (
        materialized_total
        != EXPECTED_RECORD_COUNT
    ):
        raise ValueError(
            "Materialized split records do not "
            "match expected dataset size.\n"
            f"Expected: {EXPECTED_RECORD_COUNT:,}\n"
            f"Found: {materialized_total:,}"
        )

    # --------------------------------------------------------
    # Create source provenance report
    # --------------------------------------------------------

    source_report = []

    for source_file in normalized_files:

        source_name = source_file.name

        source_report.append(
            {
                "source_file": source_name,
                "records": (
                    source_record_counts[
                        source_name
                    ]
                ),
                "train_records": (
                    source_split_counts[
                        source_name
                    ]["train"]
                ),
                "validation_records": (
                    source_split_counts[
                        source_name
                    ]["validation"]
                ),
                "test_records": (
                    source_split_counts[
                        source_name
                    ]["test"]
                ),
            }
        )

    source_report_path = (
        REPORT_DIR
        / "materialization_source_distribution.csv"
    )

    pd.DataFrame(
        source_report
    ).to_csv(
        source_report_path,
        index=False,
    )

    # --------------------------------------------------------
    # Create materialization summary
    # --------------------------------------------------------

    summary = {
        "status": "PASS",
        "random_state": RANDOM_STATE,
        "chunk_size": CHUNK_SIZE,
        "normalized_source_files": len(
            normalized_files
        ),
        "normalized_source_records": (
            total_rows_processed
        ),
        "expected_feature_groups": (
            EXPECTED_GROUP_COUNT
        ),
        "expected_records": (
            EXPECTED_RECORD_COUNT
        ),
        "materialized_records": (
            materialized_total
        ),
        "source_column_count": (
            EXPECTED_SOURCE_COLUMN_COUNT
        ),
        "output_column_count": (
            EXPECTED_OUTPUT_COLUMN_COUNT
        ),
        "traffic_feature_count": (
            EXPECTED_FEATURE_COUNT
        ),
        "preserved_columns": [
            "78 traffic features",
            LABEL_COLUMN,
            RAW_LABEL_COLUMN,
            SOURCE_COLUMN,
        ],
        "excluded_from_ml_features": [
            "feature_group_id",
            SOURCE_COLUMN,
        ],
        "split_outputs": (
            split_validation
        ),
        "source_distribution_report": str(
            source_report_path.relative_to(
                PROJECT_ROOT
            )
        ),
    }

    summary_path = (
        REPORT_DIR
        / "materialization_summary.json"
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

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("=" * 70)
    print(
        "MATERIALIZATION COMPLETE"
    )
    print("=" * 70)

    print()

    for split in EXPECTED_SPLITS:

        info = split_validation[
            split
        ]

        print(
            f"{split:<12} "
            f"{info['records']:>10,} records "
            f"{info['file_size_mb']:>10.2f} MB"
        )

    print()

    print(
        f"Total records: "
        f"{materialized_total:,}"
    )

    print(
        f"Expected records: "
        f"{EXPECTED_RECORD_COUNT:,}"
    )

    print()

    print(
        "Output files:"
    )

    for split in EXPECTED_SPLITS:
        print(
            f"  {output_paths[split]}"
        )

    print()

    print(
        "Source provenance report:"
    )

    print(
        f"  {source_report_path}"
    )

    print()

    print(
        "Materialization summary:"
    )

    print(
        f"  {summary_path}"
    )

    print()

    print(
        "STATUS: PASS"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    materialize()