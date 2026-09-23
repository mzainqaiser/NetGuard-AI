from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"F:\Projects\NetGuard-AI")

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    print("=" * 70)
    print("NetGuard AI — Processed Label Inspection")
    print("Phase 2, Task 10-A")
    print("=" * 70)

    files = sorted(PROCESSED_DIR.glob("*.csv"))

    if len(files) != 8:
        raise FileNotFoundError(
            f"Expected 8 processed CSV files, found {len(files)}."
        )

    label_counts = {}

    for file_path in files:

        print(f"\nReading: {file_path.name}")

        df = pd.read_csv(
            file_path,
            usecols=["Label"]
        )

        labels = df["Label"].dropna()

        print(f"Rows: {len(labels):,}")
        print("Unique labels:")

        for label in sorted(labels.unique()):
            print(f"  {repr(label)}")

        file_counts = labels.value_counts()

        for label, count in file_counts.items():
            label_counts[label] = (
                label_counts.get(label, 0) + int(count)
            )

    # ---------------------------------------------------------
    # Consolidated label distribution
    # ---------------------------------------------------------

    label_distribution = pd.DataFrame(
        [
            {
                "raw_label": label,
                "record_count": count,
            }
            for label, count in label_counts.items()
        ]
    )

    label_distribution = label_distribution.sort_values(
        by="record_count",
        ascending=False
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        REPORTS_DIR
        / "processed_label_inspection.csv"
    )

    label_distribution.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("LABEL INSPECTION COMPLETED")
    print("=" * 70)

    print(f"\nTotal unique labels: {len(label_distribution)}")
    print(
        f"Total records: "
        f"{label_distribution['record_count'].sum():,}"
    )

    print("\nConsolidated label distribution:")

    for _, row in label_distribution.iterrows():
        print(
            f"  {repr(row['raw_label'])}: "
            f"{row['record_count']:,}"
        )

    print(f"\nReport created:")
    print(output_path)

    print("\nProcessed CSV files were NOT modified.")


if __name__ == "__main__":
    main()