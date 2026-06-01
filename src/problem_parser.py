"""
problem_parser.py — Extracts problem scores from HTML description files.

Handles both English ("Score") and Japanese ("配点") descriptions.
Only reads files matching the provided problem_ids list.
"""
import re
from pathlib import Path
import polars as pl


def extract_problem_scores(
    descriptions_dir: str | Path,
    problem_ids: list[str]
) -> pl.DataFrame:
    """
    Extracts the score (points) from HTML problem description files,
    reading ONLY the files matching the provided problem_ids.

    Args:
        descriptions_dir: Directory containing HTML files (one per problem).
        problem_ids:      List of IDs to process (e.g. ['p02534', 'p02535']).

    Returns:
        DataFrame with columns: problem_id, score (nulls excluded).
    """
    # Matches "Score" or "配点" (Japanese), followed by a <var> tag with digits
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
            print(f"  Warning: could not read {filepath.name}: {e}")

    print(f"  HTML files targeted  : {len(problem_ids)}")
    print(f"  Files found          : {files_scanned}")
    print(f"  Scores extracted     : {files_with_score}")
    print(f"  Without score        : {files_scanned - files_with_score}")

    return pl.DataFrame(data) if data else pl.DataFrame({"problem_id": [], "score": []})
