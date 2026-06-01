"""
data_loader.py — Single point of access for all raw data on disk.

Single responsibility: know where files are and load them.
No business logic here — other modules receive DataFrames, not file paths.
"""
from pathlib import Path
import polars as pl

# ── Path configuration ─────────────────────────────────────────────────────────
RAW_DATA_DIR       = Path("./data/Project_CodeNet")
PROCESSED_DATA_DIR = Path("./data/processed")
METADATA_DIR       = RAW_DATA_DIR / "metadata"
DESCRIPTIONS_DIR   = RAW_DATA_DIR / "problem_descriptions"

# ── Pipeline output paths ──────────────────────────────────────────────────────
OUT_ABC_PROBLEMS     = PROCESSED_DATA_DIR / "atcoder_abc_problems_labeled.csv"
OUT_USER_PROFILES    = PROCESSED_DATA_DIR / "atcoder_user_profiles.csv"
OUT_USERS_CLASSIFIED = PROCESSED_DATA_DIR / "atcoder_users_classified.csv"


def load_problem_list() -> pl.DataFrame:
    """Returns the full problem list (AtCoder + AIZU)."""
    return pl.read_csv(METADATA_DIR / "problem_list.csv")


def load_atcoder_problems() -> pl.DataFrame:
    """Returns AtCoder problems only."""
    return load_problem_list().filter(pl.col("dataset") == "AtCoder")


def load_submissions_lazy(
    problem_ids: list[str],
    columns: list[str] | None = None
) -> pl.LazyFrame:
    """
    Returns a LazyFrame of submissions for the given problem IDs.

    Args:
        problem_ids: List of IDs to load (e.g. ['p02534', 'p02535', ...])
        columns:     Subset of columns to load (None = all).
                     Available: submission_id, problem_id, user_id,
                     date, language, original_language, filename_ext,
                     status, cpu_time, memory, code_size, accuracy.

    Raises:
        ValueError if no valid submission file is found.
    """
    default_dtypes = {
        "problem_id": pl.Utf8,
        "user_id":    pl.Utf8,
        "status":     pl.Utf8,
        "date":       pl.Utf8,
        "language":   pl.Utf8,
    }

    lazy_frames = []
    for pid in problem_ids:
        filepath = METADATA_DIR / f"{pid}.csv"
        if filepath.exists() and filepath.stat().st_size > 50:
            try:
                lf = pl.scan_csv(str(filepath), schema_overrides=default_dtypes)
                if columns:
                    lf = lf.select(columns)
                lazy_frames.append(lf)
            except Exception:
                pass

    if not lazy_frames:
        raise ValueError(
            f"No valid submission file found for {len(problem_ids)} IDs. "
            f"Check path: {METADATA_DIR}"
        )

    return pl.concat(lazy_frames)


def get_descriptions_dir() -> Path:
    """Returns the path to the HTML problem description files."""
    return DESCRIPTIONS_DIR


def ensure_processed_dir() -> Path:
    """Creates the processed/ directory if needed and returns its path."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DATA_DIR
