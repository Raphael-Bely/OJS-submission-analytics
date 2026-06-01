"""
user_profiling.py — Construction des profils utilisateurs et classification G1–G6.
                    Builds user profiles and classifies users into groups G1–G6.

Méthodologie : Shimizu et al. (2025).
Chaque utilisateur est classé selon la lettre de difficulté maximale
qu'il a résolue avec succès (Accepted) dans les contests ABC.
  G1 = max A | G2 = max B | G3 = max C
  G4 = max D | G5 = max E | G6 = max F

Methodology: Shimizu et al. (2025).
Each user is classified by the maximum difficulty letter they successfully
solved (Accepted) in ABC contests.
"""
import polars as pl

from difficulty_labeler import LETTER_TO_GROUP


def build_user_profiles(
    lazy_subs: pl.LazyFrame,
    df_abc: pl.DataFrame,
) -> pl.DataFrame:
    """
    Calcule les profils comportementaux de chaque utilisateur
    et les classe en groupes G1–G6 (Shimizu et al. 2025).

    Computes behavioral profiles for each user
    and classifies them into groups G1–G6 (Shimizu et al. 2025).

    Args:
        lazy_subs : LazyFrame des soumissions ABC
                    (colonnes : problem_id, user_id, status, date)
        df_abc    : DataFrame des problèmes ABC labellisés
                    (colonnes : problem_id, difficulty, ...)

    Returns:
        DataFrame par utilisateur avec :
        user_id, unique_problems_attempted, unique_problems_solved,
        total_raw_submissions, problem_win_rate, submissions_per_problem,
        max_difficulty_solved, proficiency_group
    """
    print("  [1/4] Jointure soumissions × labels de difficulté...")
    lazy_abc = df_abc.lazy().select(["problem_id", "difficulty"])

    joined = lazy_subs.join(lazy_abc, on="problem_id", how="inner")

    # ── Étape 2 : agrégation par (user, problème) ──────────────────────────────
    # Évite de compter le spam de soumissions comme des tentatives distinctes.
    # Avoids counting submission spam as distinct attempts.
    print("  [2/4] Agrégation par (utilisateur, problème)...")
    user_problem_stats = joined.group_by(["user_id", "problem_id"]).agg(
        pl.len().alias("attempts"),
        (pl.col("status") == "Accepted").any().alias("is_solved"),
        pl.col("difficulty").first().alias("difficulty"),
    )

    # ── Étape 3 : profil global par utilisateur ────────────────────────────────
    print("  [3/4] Calcul des profils globaux...")
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

    # ── Étape 4 : lettre max résolue → groupe G1–G6 ───────────────────────────
    # max() sur 'A'..'F' est correct alphabétiquement.
    # max() on 'A'..'F' is correct alphabetically.
    print("  [4/4] Classification G1–G6 (lettre max résolue)...")
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

    # Filtrage : ≥ 3 soumissions brutes (critère Shimizu et al. 2025)
    before = df_final.shape[0]
    df_final = df_final.filter(pl.col("total_raw_submissions") >= 3)
    removed = before - df_final.shape[0]

    print(f"\n  Profils calculés       : {before:,}")
    print(f"  Profils fantômes exclus (< 3 soumissions) : {removed:,}")
    print(f"  Profils actifs retenus : {df_final.shape[0]:,}")

    return df_final
