from pathlib import Path

import numpy as np
import pandas as pd


DATASET_DIR = Path(
    r"F:\Projects\NetGuard-AI\data\raw\MachineLearningCSV\MachineLearningCVE"
)


def main():
    files = sorted(DATASET_DIR.glob("*.csv"))

    print("=" * 110)
    print("NetGuard AI — CIC-IDS2017 Dataset Quality Check")
    print("=" * 110)
    print()

    print(f"CSV files found: {len(files)}")
    print()

    header = (
        f"{'File':55} "
        f"{'Rows':>10} "
        f"{'Missing':>10} "
        f"{'Inf Values':>12} "
        f"{'Rows w/ Inf':>12}"
    )

    print(header)
    print("-" * 110)

    for file_path in files:
        df = pd.read_csv(file_path)

        missing_values = int(df.isna().sum().sum())

        numeric_df = df.select_dtypes(include=np.number)

        infinite_values = int(
            np.isinf(numeric_df).sum().sum()
        )

        rows_with_infinity = int(
            np.isinf(numeric_df).any(axis=1).sum()
        )

        print(
            f"{file_path.name[:55]:55} "
            f"{len(df):10,} "
            f"{missing_values:10,} "
            f"{infinite_values:12,} "
            f"{rows_with_infinity:12,}"
        )

    print("-" * 110)
    print()
    print("Quality check completed.")


if __name__ == "__main__":
    main()