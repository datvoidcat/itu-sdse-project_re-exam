# Data loading module for the MLOps project
# This file contains functions to load data from files and DVC storage

import subprocess
import pandas as pd
from pathlib import Path
from loguru import logger


def pull_dvc_data():
    """Pull latest data from DVC remote storage."""
    logger.info("Pulling data from DVC...")
    try:
        subprocess.run(["dvc", "pull"], check=True)
        logger.success("DVC pull complete!")
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC pull failed: {e}")
        raise


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Load raw CSV data."""
    logger.info(f"Loading data from {filepath}")
    data = pd.read_csv(filepath)
    logger.info(f"Loaded {len(data)} rows")
    return data

# Run DVC pull if this script is executed directly

if __name__ == "__main__":
    pull_dvc_data()
