"""
difficulty_labeler.py — Assigns difficulty letters A–F to ABC problems.

Methodology: Shimizu et al. (2025).
The letter is determined by the ordinal position of the problem within
its contest in problem_list.csv (problems are already sorted by
increasing difficulty therein).
"""
from collections import defaultdict
import polars as pl


LETTERS = ['A', 'B', 'C', 'D', 'E', 'F']

# Letter to proficiency group mapping (Shimizu et al. 2025)
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
    Filters AtCoder Beginner Contest problems and assigns difficulty
    letter A–F by ordinal position within each contest.

    Args:
        df_atcoder: AtCoder problem DataFrame
                    (expected columns: id, name, dataset, ...)

    Returns:
        DataFrame with columns: problem_id, contest, difficulty,
        + original columns (name, time_limit, memory_limit, ...)
    """
    df_abc = df_atcoder.filter(
        pl.col("name").str.starts_with("AtCoder Beginner Contest")
    )

    # Group by contest (everything before the last " - ")
    contest_groups: dict[str, list[str]] = defaultdict(list)
    for row in df_abc.iter_rows(named=True):
        contest = row["name"].rsplit(" - ", 1)[0]
        contest_groups[contest].append(row["id"])

    # Assign letters by position
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

    # Summary
    sizes: dict[int, int] = defaultdict(int)
    for ids in contest_groups.values():
        sizes[len(ids)] += 1

    print(f"  ABC contests found   : {len(contest_groups)}")
    print(f"  Problems labeled     : {len(labeled_rows)}")
    for size in sorted(sizes):
        print(f"    {size} problems/contest ({', '.join(LETTERS[:size])}) : {sizes[size]} contests")

    print("\n  Problems per difficulty letter:")
    difficulty_counts = (
        df_labeled
        .group_by("difficulty")
        .agg(pl.len().alias("n"))
        .sort("difficulty")
    )
    for row in difficulty_counts.iter_rows(named=True):
        print(f"    {row['difficulty']} : {row['n']} problems")

    # Join back with original columns (rename id → problem_id for consistency)
    return df_labeled.join(
        df_abc.rename({"id": "problem_id"}),
        on="problem_id",
        how="left"
    )
