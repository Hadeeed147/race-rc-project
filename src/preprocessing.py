"""
RACE preprocessing — Session 1.
Loads data/train.csv, performs an 80/10/10 split (random_state=42),
and saves data/train_split.csv, data/val_split.csv, data/test_split.csv.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
INPUT_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_OUT = os.path.join(DATA_DIR, "train_split.csv")
VAL_OUT = os.path.join(DATA_DIR, "val_split.csv")
TEST_OUT = os.path.join(DATA_DIR, "test_split.csv")


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found.")
        print("Place the RACE train.csv in the data/ folder and re-run.")
        sys.exit(1)

    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    print("\nSplit complete.")
    print(f"  train_split.csv : {len(train_df):>6} rows  -> {TRAIN_OUT}")
    print(f"  val_split.csv   : {len(val_df):>6} rows  -> {VAL_OUT}")
    print(f"  test_split.csv  : {len(test_df):>6} rows  -> {TEST_OUT}")


if __name__ == "__main__":
    main()
