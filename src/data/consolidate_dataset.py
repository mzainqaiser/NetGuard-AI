from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "normalized"
OUTPUT_DIR = PROJECT_ROOT / "data" / "consolidated"
OUTPUT_FILE = OUTPUT_DIR / "netguard_dataset.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INITIALIZATION
# ============================================================

files = sorted(INPUT_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError(
        f"No normalized CSV files found in: {INPUT_DIR}"
    )

print("=" * 70)
print("TASK 12-B: DATASET CONSOLIDATION")
print("=" * 70)

print(f"Input directory:  {INPUT_DIR}")
print(f"Output file:      {OUTPUT_FILE}")
print(f"Files to combine: {len(files)}")
print()


# ============================================================
# REMOVE PREVIOUS OUTPUT IF IT EXISTS
# ============================================================

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()
    print("Previous consolidated dataset removed.")
    print()


# ============================================================
# CONSOLIDATE FILES ONE AT A TIME
# ============================================================

writer = None
total_rows = 0
reference_columns = None

try:

    for file_path in files:

        print(f"Reading: {file_path.name}")

        df = pd.read_csv(
            file_path,
            low_memory=False
        )

        # ----------------------------------------------------
        # Validate basic structure
        # ----------------------------------------------------

        if reference_columns is None:
            reference_columns = list(df.columns)
        else:
            if list(df.columns) != reference_columns:
                raise ValueError(
                    f"Column structure mismatch in "
                    f"{file_path.name}"
                )

        # ----------------------------------------------------
        # Validate labels
        # ----------------------------------------------------

        if "Label" not in df.columns:
            raise ValueError(
                f"Label column missing in {file_path.name}"
            )

        if "Raw_Label" not in df.columns:
            raise ValueError(
                f"Raw_Label column missing in {file_path.name}"
            )

        # ----------------------------------------------------
        # Validate missing values
        # ----------------------------------------------------

        missing_values = int(
            df.isna().sum().sum()
        )

        if missing_values != 0:
            raise ValueError(
                f"{file_path.name} contains "
                f"{missing_values} missing values."
            )

        # ----------------------------------------------------
        # Validate infinite values
        # ----------------------------------------------------

        numeric_columns = df.select_dtypes(
            include="number"
        )

        infinite_values = int(
            numeric_columns.isin(
                [float("inf"), float("-inf")]
            ).sum().sum()
        )

        if infinite_values != 0:
            raise ValueError(
                f"{file_path.name} contains "
                f"{infinite_values} infinite values."
            )

        # ----------------------------------------------------
        # Convert pandas DataFrame to Arrow Table
        # ----------------------------------------------------

        table = pa.Table.from_pandas(
            df,
            preserve_index=False
        )

        # ----------------------------------------------------
        # Initialize Parquet writer
        # ----------------------------------------------------

        if writer is None:
            writer = pq.ParquetWriter(
                OUTPUT_FILE,
                table.schema,
                compression="snappy"
            )

        # ----------------------------------------------------
        # Verify schema consistency
        # ----------------------------------------------------

        if not table.schema.equals(
            writer.schema,
            check_metadata=False
        ):
            raise ValueError(
                f"Parquet schema mismatch in "
                f"{file_path.name}"
            )

        # ----------------------------------------------------
        # Write batch
        # ----------------------------------------------------

        writer.write_table(table)

        total_rows += len(df)

        print(
            f"  Rows written: {len(df):,}"
        )

        print(
            f"  Total rows:    {total_rows:,}"
        )

        print()


finally:

    if writer is not None:
        writer.close()


# ============================================================
# FINAL VALIDATION
# ============================================================

print("=" * 70)
print("CONSOLIDATION COMPLETED")
print("=" * 70)

parquet_file = pq.ParquetFile(OUTPUT_FILE)

row_count = parquet_file.metadata.num_rows
column_count = parquet_file.metadata.num_columns

print(f"Parquet rows:    {row_count:,}")
print(f"Parquet columns: {column_count}")
print(f"Expected rows:   {total_rows:,}")
print(f"Output file:     {OUTPUT_FILE}")

# ------------------------------------------------------------
# Verify row count
# ------------------------------------------------------------

if row_count != total_rows:
    raise ValueError(
        "Final Parquet row count does not match "
        "the number of rows written."
    )

# ------------------------------------------------------------
# Verify column count
# ------------------------------------------------------------

if column_count != len(reference_columns):
    raise ValueError(
        "Final Parquet column count does not match "
        "the source dataset."
    )

# ------------------------------------------------------------
# Verify output file exists
# ------------------------------------------------------------

if not OUTPUT_FILE.exists():
    raise FileNotFoundError(
        "Consolidated Parquet file was not created."
    )

file_size_mb = (
    OUTPUT_FILE.stat().st_size / (1024 * 1024)
)

print(
    f"File size:       {file_size_mb:.2f} MB"
)

print()
print("Verification:")
print("  All source files processed.")
print("  Column structures were consistent.")
print("  Missing values: 0")
print("  Infinite values: 0")
print("  Row count preserved.")
print("  Source normalized CSV files were not modified.")
print()
print("STATUS: PASS")
print("=" * 70)