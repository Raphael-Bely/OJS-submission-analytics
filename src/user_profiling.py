"""
user_profiling.py — Builds user behavioral profiles and classifies them into G1–G6.

Methodology: Shimizu et al. (2025).
Each user is classified by the maximum difficulty letter they successfully
solved (Accepted) in ABC contests:
  G1 = max A | G2 = max B | G3 = max C
  G4 = max D | G5 = max E | G6 = max F
"""
import polars as pl

from difficulty_labeler import LETTER_TO_GROUP


def build_user_profiles(
    lazy_subs: pl.LazyFrame,
    df_abc: pl.DataFrame,
) -> pl.DataFrame:
    """
    Computes behavioral profiles for each user and classifies them
    into groups G1–G6 (Shimizu et al. 2025).

    Args:
        lazy_subs: LazyFrame of ABC submissions
                   (required columns: problem_id, user_id, status, date)
        df_abc:    Labeled ABC problem DataFrame
                   (required columns: problem_id, difficulty, ...)

    Returns:
        Per-user DataFrame with:
        user_id, unique_problems_attempted, unique_problems_solved,
        total_raw_submissions, problem_win_rate, submissions_per_problem,
        max_difficulty_solved, proficiency_group
    """
    print("  [1/4] Joining submissions with difficulty labels...")
    lazy_abc = df_abc.lazy().select(["problem_id", "difficulty"])
    joined = lazy_subs.join(lazy_abc, on="problem_id", how="inner")

    # Step 2: aggregate per (user, problem) to handle submission spam
    print("  [2/4] Aggregating per (user, problem)...")
    user_problem_stats = joined.group_by(["user_id", "problem_id"]).agg(
        pl.len().alias("attempts"),
        (pl.col("status") == "Accepted").any().alias("is_solved"),
        pl.col("difficulty").first().alias("difficulty"),
    )

    # Step 3: global profile per user
    print("  [3/4] Computing global user profiles...")
    user_profiles = (
        user_problem_stats
        .group_by("user_id")
        .agg(
            pl.len().alias("unique_problems_attempted"),
            pl.col("is_solved").sum().alias("unique_problems_solved"),
            pl.col("attempts").sum().alias("total_raw_submissions"),
        )
        .with_columns([
            (pl.col("unique_problems_solved") / pl.col("unique_problems_attempted"))
              .alias("problem_win_rate"),
            (pl.col("total_raw_submissions") / pl.col("unique_problems_attempted"))
              .alias("submissions_per_problem"),
        ])
    )

    # Step 4: max solved difficulty letter → G1–G6 group
    # max() on 'A'..'F' is alphabetically correct (A < B < C < D < E < F)
    print("  [4/4] Classifying into G1–G6 (max solved difficulty letter)...")
    max_difficulty = (
        user_problem_stats
        .filter(pl.col("is_solved"))
        .group_by("user_id")
        .agg(pl.col("difficulty").max().alias("max_difficulty_solved"))
    )

    user_profiles = user_profiles.join(max_difficulty, on="user_id", how="left")
    user_profiles = user_profiles.with_columns(
        pl.col("max_difficulty_solved")
          .replace(LETTER_TO_GROUP)
          .alias("proficiency_group")
    )

    df_final = user_profiles.collect()

    # Filter: >= 3 raw submissions (Shimizu et al. 2025 criterion)
    before = df_final.shape[0]
    df_final = df_final.filter(pl.col("total_raw_submissions") >= 3)
    removed = before - df_final.shape[0]

    print(f"\n  Profiles computed          : {before:,}")
    print(f"  Ghost profiles removed     : {removed:,}  (< 3 submissions)")
    print(f"  Active profiles kept       : {df_final.shape[0]:,}")

    return df_final
