import polars as pl
from pathlib import Path
import re

def extract_problem_scores(descriptions_dir: str | Path, problem_ids: list[str]) -> pl.DataFrame:
    """
    Extrait les scores (points) depuis les fichiers HTML de description,
    en ne lisant QUE les fichiers correspondant aux problem_ids fournis.
    Gère les descriptions en anglais et en japonais.

    Extracts scores (points) from HTML description files,
    reading ONLY the files matching the provided problem_ids.
    Handles descriptions in English and Japanese.

    Args:
        descriptions_dir : dossier contenant les fichiers HTML (un par problème).
        problem_ids      : liste des IDs à traiter (ex: ['p02534', 'p02535', ...]).

    Returns:
        DataFrame avec colonnes : problem_id, score (nulls exclus).
    """
    # Cherche "Score" ou "配点" (japonais), suivi de la balise <var> avec des chiffres
    score_pattern = re.compile(
        r"(?:Score|配点).*?<var>(\d+)</var>",
        re.IGNORECASE | re.DOTALL
    )

    path = Path(descriptions_dir)
    data = []
    files_scanned = 0
    files_with_score = 0

    for pid in problem_ids:
        filepath = path / f"{pid}.html"
        if not filepath.exists():
            continue

        files_scanned += 1
        try:
            content = filepath.read_text(encoding="utf-8")
            match = score_pattern.search(content)
            if match:
                data.append({"problem_id": pid, "score": int(match.group(1))})
                files_with_score += 1
        except Exception as e:
            print(f"  ⚠ Erreur lecture {filepath.name}: {e}")

    print(f"  Fichiers HTML ciblés   : {len(problem_ids)}")
    print(f"  Fichiers trouvés       : {files_scanned}")
    print(f"  Scores extraits        : {files_with_score}")
    print(f"  Sans score (format old): {files_scanned - files_with_score}")

    return pl.DataFrame(data) if data else pl.DataFrame({"problem_id": [], "score": []})