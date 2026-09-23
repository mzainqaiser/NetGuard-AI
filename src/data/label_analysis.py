"""
NetGuard AI — CIC-IDS2017 Label Analysis
Phase 2, Task 07 — Dataset Acquisition & Initial Inspection

Purpose
-------
Analyze the Label column across all 8 CIC-IDS2017 CSV files:

    - Per-file label counts and percentages
    - Overall combined label distribution
    - Label-name consistency checks
    - Empty/missing label detection
    - Leading/trailing whitespace detection
    - Dataset completeness verification

This script is READ-ONLY with respect to the source dataset.

It does NOT perform:
    - preprocessing
    - feature engineering
    - label normalization
    - encoding
    - class balancing
    - deduplication
    - model training

Only summary reports are written to the separate reports directory.
"""

from pathlib import Path

import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(r"F:\Projects\NetGuard-AI")

# Folder containing the CIC-IDS2017 CSV files.
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "MachineLearningCSV" / "MachineLearningCVE"

# Folder where analysis reports will be saved.
REPORTS_DIR = PROJECT_ROOT / "reports"


# Official CIC-IDS2017 CSV filenames expected for this project.
EXPECTED_FILES = {
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}


# CIC-IDS2017 may contain the label column with or without a leading space.
POSSIBLE_LABEL_COLUMNS = ["Label", " Label"]


# ============================================================================
# DATASET DISCOVERY
# ============================================================================

def discover_csv_files(dataset_dir: Path) -> list[Path]:
    """
    Discover CSV files in the dataset directory.

    Only the directory itself is searched; subdirectories are not searched.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    if not dataset_dir.is_dir():
        raise NotADirectoryError(
            f"Dataset path is not a directory: {dataset_dir}"
        )

    return sorted(dataset_dir.glob("*.csv"))


# ============================================================================
# LABEL COLUMN DETECTION
# ============================================================================

def find_label_column(csv_path: Path) -> str | None:
    """
    Read only the CSV header and identify the label column.

    Supports:
        Label
        " Label"
    """
    header = pd.read_csv(csv_path, nrows=0)

    for candidate in POSSIBLE_LABEL_COLUMNS:
        if candidate in header.columns:
            return candidate

    return None


# ============================================================================
# SINGLE-FILE ANALYSIS
# ============================================================================

def analyze_single_file(csv_path: Path) -> dict:
    """
    Analyze the Label column of one CIC-IDS2017 CSV file.

    Only the Label column is loaded into memory.

    The original CSV is never modified.
    """

    result = {
        "filename": csv_path.name,
        "status": "ok",
        "error": None,
        "label_column_used": None,
        "total_rows": 0,
        "unique_labels": 0,
        "label_counts": {},
        "has_empty_labels": False,
        "empty_label_count": 0,
        "has_whitespace_issue": False,
        "whitespace_labels": set(),
    }

    try:
        # ------------------------------------------------------------
        # Identify Label column
        # ------------------------------------------------------------

        label_col = find_label_column(csv_path)

        if label_col is None:
            result["status"] = "missing_label_column"
            result["error"] = (
                "No 'Label' or ' Label' column found in the header."
            )
            return result

        result["label_column_used"] = label_col

        # ------------------------------------------------------------
        # Read ONLY the Label column
        # ------------------------------------------------------------

        df = pd.read_csv(
            csv_path,
            usecols=[label_col],
            dtype=str,
            keep_default_na=False,
            na_values=[],
        )

        result["total_rows"] = len(df)

        raw_series = df[label_col]

        # ------------------------------------------------------------
        # Empty / blank label detection
        # ------------------------------------------------------------

        is_blank = raw_series.str.strip() == ""

        result["empty_label_count"] = int(is_blank.sum())
        result["has_empty_labels"] = result["empty_label_count"] > 0

        # ------------------------------------------------------------
        # Leading / trailing whitespace detection
        # ------------------------------------------------------------

        has_leading_trailing_space = (
            (raw_series != raw_series.str.strip())
            & (~is_blank)
        )

        whitespace_values = set(
            raw_series[has_leading_trailing_space].unique()
        )

        result["has_whitespace_issue"] = len(whitespace_values) > 0
        result["whitespace_labels"] = whitespace_values

        # ------------------------------------------------------------
        # Raw label distribution
        # ------------------------------------------------------------

        counts = raw_series.value_counts(dropna=False)

        result["unique_labels"] = int(counts.shape[0])
        result["label_counts"] = counts.to_dict()

    except Exception as exc:
        result["status"] = "read_error"
        result["error"] = str(exc)

    return result


# ============================================================================
# TERMINAL REPORT — SINGLE FILE
# ============================================================================

def print_file_report(result: dict) -> None:
    """Print the analysis results for one file."""

    print(f"File: {result['filename']}")

    if result["status"] != "ok":
        print(f"  STATUS: {result['status']}")
        print(f"  ERROR:  {result['error']}")
        print("-" * 70)
        return

    total = result["total_rows"]

    print(f"Total Rows: {total}")
    print(f"Unique Labels: {result['unique_labels']}")

    print("\nLabel Distribution:")

    for label, count in sorted(
        result["label_counts"].items(),
        key=lambda item: -item[1],
    ):
        percentage = (
            count / total * 100
            if total
            else 0.0
        )

        display_label = (
            label if label != "" else "<EMPTY>"
        )

        print(
            f"  {display_label:<35}"
            f"{count:>12,}"
            f"    {percentage:7.2f}%"
        )

    if result["has_empty_labels"]:
        print(
            f"\n  [!] Empty/blank labels: "
            f"{result['empty_label_count']:,}"
        )

    if result["has_whitespace_issue"]:
        whitespace_list = ", ".join(
            repr(value)
            for value in sorted(result["whitespace_labels"])
        )

        print(
            "  [!] Labels with leading/trailing "
            f"whitespace: {whitespace_list}"
        )

    print("-" * 70)


# ============================================================================
# OVERALL LABEL DISTRIBUTION
# ============================================================================

def calculate_overall_distribution(successful_results: list[dict]):
    """
    Combine label counts from successfully analyzed files.

    No raw datasets are concatenated.
    """

    overall_counts = {}
    overall_total = 0

    for result in successful_results:
        overall_total += result["total_rows"]

        for label, count in result["label_counts"].items():
            overall_counts[label] = (
                overall_counts.get(label, 0) + count
            )

    return overall_counts, overall_total


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 70)
    print("NetGuard AI — CIC-IDS2017 Label Analysis")
    print("Phase 2, Task 07 — Dataset Acquisition & Initial Inspection")
    print("=" * 70)
    print()

    # ------------------------------------------------------------
    # Create reports directory
    # ------------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Discover CSV files
    # ------------------------------------------------------------

    discovered_files = discover_csv_files(DATASET_DIR)

    if not discovered_files:
        print(
            f"No CSV files found in:\n{DATASET_DIR}"
        )
        return

    discovered_names = {
        file.name for file in discovered_files
    }

    unexpected_files = (
        discovered_names - EXPECTED_FILES
    )

    missing_expected_files = (
        EXPECTED_FILES - discovered_names
    )

    expected_files_found = (
        discovered_names & EXPECTED_FILES
    )

    # ------------------------------------------------------------
    # Only analyze official expected files
    # ------------------------------------------------------------

    files_to_analyze = sorted(
        file
        for file in discovered_files
        if file.name in EXPECTED_FILES
    )

    # ------------------------------------------------------------
    # Dataset discovery summary
    # ------------------------------------------------------------

    print("DATASET DISCOVERY")
    print("-" * 70)

    print(
        f"CSV files discovered:        "
        f"{len(discovered_files)}"
    )

    print(
        f"Expected CIC-IDS2017 files:  "
        f"{len(EXPECTED_FILES)}"
    )

    print(
        f"Expected files found:        "
        f"{len(expected_files_found)}"
    )

    print(
        f"Expected files missing:      "
        f"{len(missing_expected_files)}"
    )

    print(
        f"Unexpected CSV files:        "
        f"{len(unexpected_files)}"
    )

    print()

    if unexpected_files:
        print(
            "[NOTICE] Unexpected CSV files found "
            "and EXCLUDED from analysis:"
        )

        for filename in sorted(unexpected_files):
            print(f"    - {filename}")

        print()

    if missing_expected_files:
        print(
            "[NOTICE] Expected CIC-IDS2017 files "
            "NOT found:"
        )

        for filename in sorted(missing_expected_files):
            print(f"    - {filename}")

        print()

    # ------------------------------------------------------------
    # Stop if no expected files are available
    # ------------------------------------------------------------

    if not files_to_analyze:
        print(
            "No expected CIC-IDS2017 files were found."
        )
        print("Analysis stopped.")
        return

    print(
        f"Analyzing {len(files_to_analyze)} "
        f"expected CIC-IDS2017 file(s)..."
    )

    print()

    # ------------------------------------------------------------
    # Analyze files one at a time
    # ------------------------------------------------------------

    per_file_results = []

    for csv_path in files_to_analyze:

        result = analyze_single_file(csv_path)

        print_file_report(result)

        per_file_results.append(result)

    # ------------------------------------------------------------
    # Separate successful and failed analyses
    # ------------------------------------------------------------

    successful_results = [
        result
        for result in per_file_results
        if result["status"] == "ok"
    ]

    failed_results = [
        result
        for result in per_file_results
        if result["status"] != "ok"
    ]

    # ------------------------------------------------------------
    # Dataset completeness summary
    # ------------------------------------------------------------

    print("=" * 70)
    print("DATASET COMPLETENESS SUMMARY")
    print("=" * 70)

    print(
        f"Expected CIC-IDS2017 files:     "
        f"{len(EXPECTED_FILES)}"
    )

    print(
        f"Expected files found:           "
        f"{len(expected_files_found)}"
    )

    print(
        f"Expected files missing:         "
        f"{len(missing_expected_files)}"
    )

    print(
        f"Unexpected CSV files:            "
        f"{len(unexpected_files)}"
    )

    print(
        f"Files successfully analyzed:    "
        f"{len(successful_results)}"
    )

    print(
        f"Files failed to analyze:        "
        f"{len(failed_results)}"
    )

    print()

    # ------------------------------------------------------------
    # Determine whether complete analysis is confirmed
    # ------------------------------------------------------------

    complete_analysis = (
        len(expected_files_found) == len(EXPECTED_FILES)
        and len(successful_results) == len(EXPECTED_FILES)
        and len(failed_results) == 0
    )

    if complete_analysis:

        print(
            "SUCCESS: All 8 expected CIC-IDS2017 "
            "files were found and successfully analyzed."
        )

    elif len(expected_files_found) < len(EXPECTED_FILES):

        print(
            "WARNING: Complete 8-file analysis "
            "cannot be confirmed because one or "
            "more expected files are missing."
        )

    else:

        print(
            "WARNING: All 8 expected files were found, "
            "but complete label analysis could not be "
            "confirmed because one or more files failed "
            "to process."
        )

    print()

    # ------------------------------------------------------------
    # Overall distribution
    # ------------------------------------------------------------

    overall_counts, overall_total = (
        calculate_overall_distribution(
            successful_results
        )
    )

    print("=" * 70)
    print("OVERALL LABEL DISTRIBUTION")
    print("=" * 70)

    print(
        f"\nTotal Records Analyzed: "
        f"{overall_total:,}\n"
    )

    print(
        f"{'Label':<35}"
        f"{'Count':>12}"
        f"    {'Percentage':>10}"
    )

    print("-" * 70)

    for label, count in sorted(
        overall_counts.items(),
        key=lambda item: -item[1],
    ):

        percentage = (
            count / overall_total * 100
            if overall_total
            else 0.0
        )

        display_label = (
            label if label != "" else "<EMPTY>"
        )

        print(
            f"{display_label:<35}"
            f"{count:>12,}"
            f"    {percentage:9.2f}%"
        )

    # ------------------------------------------------------------
    # Label consistency analysis
    # ------------------------------------------------------------

    all_raw_labels = set(
        overall_counts.keys()
    )

    normalized_map = {}

    for label in all_raw_labels:

        normalized_label = (
            label.strip().lower()
        )

        normalized_map.setdefault(
            normalized_label,
            set(),
        ).add(label)

    inconsistent_groups = {
        normalized: variants
        for normalized, variants
        in normalized_map.items()
        if len(variants) > 1
    }

    # ------------------------------------------------------------
    # Quality issue lists
    # ------------------------------------------------------------

    files_missing_label_column = [
        result["filename"]
        for result in per_file_results
        if result["status"]
        == "missing_label_column"
    ]

    files_with_read_errors = [
        result["filename"]
        for result in per_file_results
        if result["status"]
        == "read_error"
    ]

    files_with_empty_labels = [
        result["filename"]
        for result in successful_results
        if result["has_empty_labels"]
    ]

    files_with_whitespace_issues = [
        result["filename"]
        for result in successful_results
        if result["has_whitespace_issue"]
    ]

    # ------------------------------------------------------------
    # Consistency / quality report
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONSISTENCY / QUALITY REPORT")
    print("=" * 70)

    print(
        f"Files successfully processed: "
        f"{len(successful_results)}"
    )

    print(
        f"Files with missing Label column: "
        f"{len(files_missing_label_column)}"
    )

    print(
        f"Files with read errors: "
        f"{len(files_with_read_errors)}"
    )

    print(
        f"Files with empty/missing labels: "
        f"{len(files_with_empty_labels)}"
    )

    print(
        f"Files with whitespace issues: "
        f"{len(files_with_whitespace_issues)}"
    )

    print(
        f"\nDistinct raw labels found "
        f"across successfully analyzed files "
        f"({len(all_raw_labels)}):"
    )

    for label in sorted(all_raw_labels):
        print(f"    - {repr(label)}")

    if inconsistent_groups:

        print(
            "\n[!] Possible label naming "
            "inconsistencies:"
        )

        for variants in inconsistent_groups.values():
            print(
                f"    - {sorted(variants)}"
            )

    else:

        print(
            "\nNo label naming inconsistencies "
            "detected after case/whitespace comparison."
        )

    # ------------------------------------------------------------
    # Save reports
    # ------------------------------------------------------------

    save_reports(
        per_file_results=per_file_results,
        overall_counts=overall_counts,
        overall_total=overall_total,
        inconsistent_groups=inconsistent_groups,
        files_missing_label_column=files_missing_label_column,
        files_with_read_errors=files_with_read_errors,
        files_with_empty_labels=files_with_empty_labels,
        files_with_whitespace_issues=files_with_whitespace_issues,
        all_raw_labels=all_raw_labels,
        unexpected_files=unexpected_files,
        missing_expected_files=missing_expected_files,
        expected_files_found=expected_files_found,
        complete_analysis=complete_analysis,
    )

    print()
    print("=" * 70)
    print("REPORTS")
    print("=" * 70)

    print(
        f"Reports saved to:\n{REPORTS_DIR}"
    )

    print()
    print("Generated files:")

    print(
        "  - label_distribution_by_file.csv"
    )

    print(
        "  - overall_label_distribution.csv"
    )

    print(
        "  - label_quality_report.txt"
    )

    print()
    print("=" * 70)
    print("Task 07 label analysis completed.")
    print("=" * 70)


# ============================================================================
# SAVE REPORTS
# ============================================================================

def save_reports(
    per_file_results,
    overall_counts,
    overall_total,
    inconsistent_groups,
    files_missing_label_column,
    files_with_read_errors,
    files_with_empty_labels,
    files_with_whitespace_issues,
    all_raw_labels,
    unexpected_files,
    missing_expected_files,
    expected_files_found,
    complete_analysis,
) -> None:
    """
    Save analysis results to the reports directory.

    These files are separate from the source dataset.
    """

    # ------------------------------------------------------------------------
    # 1. Per-file label distribution
    # ------------------------------------------------------------------------

    per_file_rows = []

    for result in per_file_results:

        if result["status"] != "ok":
            continue

        total = result["total_rows"]

        for label, count in result["label_counts"].items():

            percentage = (
                count / total * 100
                if total
                else 0.0
            )

            per_file_rows.append(
                {
                    "filename": result["filename"],
                    "label": label,
                    "count": count,
                    "percentage": round(
                        percentage,
                        4,
                    ),
                    "file_total_rows": total,
                }
            )

    pd.DataFrame(
        per_file_rows
    ).to_csv(
        REPORTS_DIR
        / "label_distribution_by_file.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # 2. Overall label distribution
    # ------------------------------------------------------------------------

    overall_rows = []

    for label, count in sorted(
        overall_counts.items(),
        key=lambda item: -item[1],
    ):

        percentage = (
            count / overall_total * 100
            if overall_total
            else 0.0
        )

        overall_rows.append(
            {
                "label": label,
                "count": count,
                "percentage": round(
                    percentage,
                    4,
                ),
            }
        )

    pd.DataFrame(
        overall_rows
    ).to_csv(
        REPORTS_DIR
        / "overall_label_distribution.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # 3. Quality report
    # ------------------------------------------------------------------------

    lines = []

    lines.append(
        "NetGuard AI — CIC-IDS2017 Label Quality Report"
    )

    lines.append("=" * 70)

    lines.append(
        f"Expected CIC-IDS2017 files: {len(EXPECTED_FILES)}"
    )

    lines.append(
        f"Expected files found: {len(expected_files_found)}"
    )

    lines.append(
        f"Expected files missing: "
        f"{len(missing_expected_files)}"
    )

    lines.append(
        f"Unexpected CSV files: "
        f"{len(unexpected_files)}"
    )

    lines.append(
        f"Files successfully analyzed: "
        f"{len([r for r in per_file_results if r['status'] == 'ok'])}"
    )

    lines.append(
        f"Files failed to analyze: "
        f"{len([r for r in per_file_results if r['status'] != 'ok'])}"
    )

    lines.append("")

    if complete_analysis:

        lines.append(
            "STATUS: All 8 expected CIC-IDS2017 "
            "files were found and successfully analyzed."
        )

    elif missing_expected_files:

        lines.append(
            "STATUS: INCOMPLETE — one or more expected "
            "CIC-IDS2017 files are missing."
        )

    else:

        lines.append(
            "STATUS: INCOMPLETE — one or more expected "
            "files failed to process."
        )

    lines.append("")

    # Missing files

    lines.append(
        "Missing expected files:"
    )

    if missing_expected_files:

        for filename in sorted(
            missing_expected_files
        ):
            lines.append(
                f"  - {filename}"
            )

    else:

        lines.append("  None")

    lines.append("")

    # Unexpected files

    lines.append(
        "Unexpected CSV files excluded from analysis:"
    )

    if unexpected_files:

        for filename in sorted(
            unexpected_files
        ):
            lines.append(
                f"  - {filename}"
            )

    else:

        lines.append("  None")

    lines.append("")

    # Label-column problems

    lines.append(
        "Files with missing Label column:"
    )

    if files_missing_label_column:

        for filename in files_missing_label_column:
            lines.append(
                f"  - {filename}"
            )

    else:

        lines.append("  None")

    lines.append("")

    # Read errors

    lines.append(
        "Files with read errors:"
    )

    if files_with_read_errors:

        for filename in files_with_read_errors:
            lines.append(
                f"  - {filename}"
            )

    else:

        lines.append("  None")

    lines.append("")

    # Empty labels

    lines.append(
        "Files with empty/missing labels:"
    )

    if files_with_empty_labels:

        for filename in files_with_empty_labels:
            lines.append(
                f"  - {filename}"
            )

    else:

        lines.append("  None")

    lines.append("")

    # Whitespace issues

    lines.append(
        "Files with leading/trailing whitespace issues:"
    )

    if files_with_whitespace_issues:

        for filename in files_with_whitespace_issues:
            lines.append(
                f"  - {filename}"
            )

    else:

        lines.append("  None")

    lines.append("")

    # Distinct labels

    lines.append(
        f"Distinct raw labels found "
        f"({len(all_raw_labels)}):"
    )

    for label in sorted(all_raw_labels):

        lines.append(
            f"  - {repr(label)}"
        )

    lines.append("")

    # Naming inconsistencies

    if inconsistent_groups:

        lines.append(
            "Possible label naming inconsistencies:"
        )

        for variants in inconsistent_groups.values():

            lines.append(
                f"  - {sorted(variants)}"
            )

    else:

        lines.append(
            "No naming inconsistencies detected "
            "after case/whitespace comparison."
        )

    # Write text report

    report_path = (
        REPORTS_DIR
        / "label_quality_report.txt"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()