from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
SUBMISSIONS_DIR = REPORTS_DIR / "submissions"


def create_output_directories() -> None:
    for directory in (
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        SUBMISSIONS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)