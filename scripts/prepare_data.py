"""
Data preparation script - Separate feature engineering from training.

This script prepares and saves preprocessed data to disk, allowing for
faster repeated training experiments without re-running feature engineering.

Usage:
    python -m scripts.prepare_data --config configs/default.yaml --output data/prepared
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data import prepare_all_data
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser(description="Prepare and save preprocessed training data")
    ap.add_argument("--config", required=True, help="Path to config file")
    ap.add_argument("--output", required=True, help="Output directory for prepared data")
    args = ap.parse_args()

    # Load config
    cfg = load_config(args.config)
    set_seed(cfg.random_seed)

    log = get_logger("prepare_data")

    output_dir = Path(args.output)
    log.info(f"Preparing data with config: {args.config}")
    log.info(f"Output directory: {output_dir}")

    # Prepare all data
    prepare_all_data(cfg, output_dir)

    log.info("Data preparation complete!")
    log.info(f"To train with prepared data, use:")
    log.info(f"  python -m src.train --config {args.config} --prepared-data {output_dir}")


if __name__ == "__main__":
    main()
