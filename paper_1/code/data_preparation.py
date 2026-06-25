"""
Data Preparation — CICIDS2017 Dataset.

Merges raw CSV files, cleans records, creates binary labels,
splits into train/test sets, and saves Parquet files for downstream
experiments.
"""

import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR, RANDOM_SEED, TEST_PATH, TRAIN_PATH
from data_utils import align_schema, load_csv_file

CSV_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

DEFAULT_RAW_DATA_DIR = os.environ.get("CICIDS2017_RAW_DIR", os.path.join(DATA_DIR, "raw"))


def merge_csv_files(input_dir: str) -> pd.DataFrame:
    """Load and merge all CICIDS2017 CSV files from a directory."""
    merged_df = None
    unified_columns = None

    print("-" * 30)
    print("START MERGING CICIDS2017 DATA")
    print("-" * 30)

    for index, filename in enumerate(CSV_FILES, start=1):
        file_path = os.path.join(input_dir, filename)
        print(f"\n[{index}/{len(CSV_FILES)}] Processing: {filename}")

        if not os.path.exists(file_path):
            print(f"  WARNING: File not found, skipping: {file_path}")
            continue

        try:
            df = load_csv_file(file_path)
            print(f"  Read successfully: {len(df):,} rows, {len(df.columns)} columns")

            if merged_df is None:
                unified_columns = df.columns.tolist()
                merged_df = df
                print(f"  Initialized schema with {len(unified_columns)} columns")
                continue

            df = align_schema(df, unified_columns)
            merged_df = pd.concat([merged_df, df], ignore_index=True)
            print("  Merged into total DataFrame")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

    if merged_df is None:
        raise FileNotFoundError(
            f"No CSV files were loaded from '{input_dir}'. "
            "Place the CICIDS2017 CSV files there or set CICIDS2017_RAW_DIR."
        )

    print("-" * 30)
    print("MERGE COMPLETED")
    print("-" * 30)
    return merged_df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean records, create binary labels, and report class distribution."""
    print(f"\nTotal rows before cleaning: {len(df):,}")
    df = df.drop_duplicates().dropna()
    print(f"Total rows after cleaning:  {len(df):,}")

    if "label" not in df.columns:
        raise KeyError("Expected a 'label' column after column-name normalisation.")

    df = df.copy()
    df["label_binary"] = (df["label"] != "BENIGN").astype(int)

    print("\nLabel distribution:")
    print(df["label"].value_counts().head(20).to_string())

    print("\nBinary label distribution:")
    print(df["label_binary"].value_counts().sort_index().to_string())

    exclude_cols = {
        "label", "label_binary", "source_ip", "destination_ip",
        "flow_id", "timestamp", "protocol",
    }
    numeric_cols = df.select_dtypes(include="number").columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    print(f"\nNumber of numeric features: {len(feature_cols)}")

    return df


def save_splits(df: pd.DataFrame, train_path: str, test_path: str) -> None:
    """Split the dataset and persist Parquet files."""
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=df["label_binary"],
    )

    print(f"\nTraining set: {len(train_df):,} samples")
    print(f"Test set:     {len(test_df):,} samples")

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    print("\nSaving to parquet...")
    train_df.to_parquet(train_path, index=False)
    print(f"  Saved: {train_path}")

    test_df.to_parquet(test_path, index=False)
    print(f"  Saved: {test_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CICIDS2017 train/test Parquet files.")
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_RAW_DATA_DIR,
        help="Directory containing raw CICIDS2017 CSV files "
             "(default: paper_1/data/raw or CICIDS2017_RAW_DIR).",
    )
    parser.add_argument("--train-path", default=TRAIN_PATH, help="Output path for training Parquet.")
    parser.add_argument("--test-path", default=TEST_PATH, help="Output path for test Parquet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    merged_df = merge_csv_files(args.input_dir)
    prepared_df = prepare_dataset(merged_df)
    save_splits(prepared_df, args.train_path, args.test_path)

    print("-" * 30)
    print("DATA PREPARATION COMPLETED")
    print("-" * 30)
    print("You can now run any experiment file, including the hybrid base model script.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Data preparation failed: {exc}", file=sys.stderr)
        sys.exit(1)
