"""
difficulty_labeler.py — Assignation des lettres de difficulté A–F aux problèmes ABC.
                        Assigns difficulty letters A–F to ABC problems.

Méthodologie : Shimizu et al. (2025).
La lettre est déterminée par la position ordinale du problème au sein de son contest
dans problem_list.csv (les problèmes y étant déjà triés par difficulté croissante).

Methodology: Shimizu et al. (2025).
The letter is determined by the ordinal position of the problem within its contest
in problem_list.csv (problems are already sorted by increasing difficulty therein).
"""
from collections import defaultdict
import polars as pl


LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']

# Correspondance lettre → groupe (Shimizu et al. 2025)
LETTER_TO_GROUP = {
    'A': 'G1',
    'B': 'G2',
    'C': 'G3',
    'D': 'G4',
    'E': 'G5',
    'F': 'G6',
}


def label_abc_problems(df_atcoder: pl.DataFrame) -> pl.DataFrame:
    """
    Filtre les problèmes AtCoder Beginner Contest et assigne la lettre A–F
    par position ordinale dans chaque contest.

    Filters AtCoder Beginner Contest problems and assigns letter A–F
    by ordinal position within each contest.

    Args:
        df_atcoder : DataFrame des problèmes AtCoder
                     (colonnes attendues : id, name, dataset, ...)

    Returns:
        DataFrame avec colonnes : problem_id, contest, difficulty,
        + colonnes originales (name, time_limit, memory_limit, ...)
    """
    df_abc = df_atcoder.filter(
        pl.col("name").str.starts_with("AtCoder Beginner Contest")
    )

    # Groupement par contest (tout ce qui précède le dernier " - ")
    contest_groups: dict[str, list[str]] = defaultdict(list)
    for row in df_abc.iter_rows(named=True):
        contest = row["name"].rsplit(" - ", 1)[0]
        contest_groups[contest].append(row["id"])

    # Assignation des lettres par position
    labeled_rows = []
    for contest, ids in sorted(contest_groups.items()):
        for i, pid in enumerate(ids):
            if i < len(LETTERS):
                labeled_rows.append({
                    "problem_id": pid,
                    "contest":    contest,
                    "difficulty": LETTERS[i],
                })

    df_labeled = pl.DataFrame(labeled_rows)

    # Bilan
    sizes: dict[int, int] = defaultdict(int)
    for ids in contest_groups.values():
        sizes[len(ids)] += 1

    print(f"  Contests ABC trouvés   : {len(contest_groups)}")
    print(f"  Problèmes labellisés   : {len(labeled_rows)}")
    for size in sorted(sizes):
        print(f"    {size} problèmes/contest ({', '.join(LETTERS[:size])}) : {sizes[size]} contests")

    print("\n  Répartition par lettre :")
    difficulty_counts = (
        df_labeled
        .group_by("difficulty")
        .agg(pl.len().alias("n"))
        .sort("difficulty")
    )
    for row in difficulty_counts.iter_rows(named=True):
        print(f"    {row['difficulty']} : {row['n']} problèmes")

    # Jointure avec les colonnes originales (rename id → problem_id pour cohérence)
    return df_labeled.join(
        df_abc.rename({"id": "problem_id"}),
        on="problem_id",
        how="left"
    )
