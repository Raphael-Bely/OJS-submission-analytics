"""
data_loader.py — Point d'accès unique aux données brutes sur disque.
               Single point of access for all raw data on disk.

Responsabilité unique : savoir où sont les fichiers et les charger.
Single responsibility : know where files are and load them.

Aucune logique métier ici. Les autres modules reçoivent des DataFrames,
pas des chemins de fichiers.

No business logic here. Other modules receive DataFrames, not file paths.
"""
from pathlib import Path
import polars as pl

# ── Configuration des chemins ──────────────────────────────────────────────────
RAW_DATA_DIR       = Path("./data/Project_CodeNet")
PROCESSED_DATA_DIR = Path("./data/processed")
METADATA_DIR       = RAW_DATA_DIR / "metadata"
DESCRIPTIONS_DIR   = RAW_DATA_DIR / "problem_descriptions"

# ── Chemins des fichiers produits (sorties du pipeline) ───────────────────────
OUT_ABC_PROBLEMS    = PROCESSED_DATA_DIR / "atcoder_abc_problems_labeled.csv"
OUT_USER_PROFILES   = PROCESSED_DATA_DIR / "atcoder_user_profiles.csv"
OUT_USERS_CLASSIFIED = PROCESSED_DATA_DIR / "atcoder_users_classified.csv"


# ── Fonctions de chargement ────────────────────────────────────────────────────

def load_problem_list() -> pl.DataFrame:
    """
    Retourne la liste complète des problèmes (AtCoder + AIZU).
    Returns the full problem list (AtCoder + AIZU).
    """
    return pl.read_csv(METADATA_DIR / "problem_list.csv")


def load_atcoder_problems() -> pl.DataFrame:
    """
    Retourne uniquement les problèmes AtCoder.
    Returns AtCoder problems only.
    """
    return load_problem_list().filter(pl.col("dataset") == "AtCoder")


def load_submissions_lazy(
    problem_ids: list[str],
    columns: list[str] | None = None
) -> pl.LazyFrame:
    """
    Retourne un LazyFrame des soumissions pour une liste de problem_ids.
    Returns a LazyFrame of submissions for the given problem IDs.

    Args:
        problem_ids : liste des IDs (ex: ['p02534', 'p02535', ...])
        columns     : sous-liste de colonnes à charger (None = toutes).
                      Colonnes disponibles : submission_id, problem_id, user_id,
                      date, language, original_language, filename_ext, status,
                      cpu_time, memory, code_size, accuracy.

    Raises:
        ValueError si aucun fichier valide n'est trouvé.
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
                lf = pl.scan_csv(str(filepath), dtypes=default_dtypes)
                if columns:
                    lf = lf.select(columns)
                lazy_frames.append(lf)
            except Exception:
                pass

    if not lazy_frames:
        raise ValueError(
            f"Aucun fichier de soumission valide trouvé pour {len(problem_ids)} IDs. "
            f"Vérifiez le chemin : {METADATA_DIR}"
        )

    return pl.concat(lazy_frames)


def get_descriptions_dir() -> Path:
    """
    Retourne le chemin vers les descriptions HTML des problèmes.
    Returns the path to problem HTML description files.
    """
    return DESCRIPTIONS_DIR


def ensure_processed_dir() -> Path:
    """
    Crée le dossier processed/ s'il n'existe pas, et le retourne.
    Creates the processed/ directory if needed and returns its path.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DATA_DIR
