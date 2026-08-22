"""
Export the 2-year fused dataset from parquet to CSV.

Usage:
    cd scripts && python export_2years_csv.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri_agent.data_access.fusion import load_fused_dataset
from agri_agent.utils.logging_config import get_logger

log = get_logger(__name__)

INPUT_PATH = "data/processed/fused_2years.parquet"
OUTPUT_PATH = "data/processed/fused_2years.csv"


def main():
    df = load_fused_dataset(INPUT_PATH)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info(
        "Exported %d rows x %d columns to %s",
        len(df), len(df.columns), OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
