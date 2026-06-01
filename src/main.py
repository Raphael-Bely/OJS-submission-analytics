"""
main.py — Orchestrateur du pipeline de données.
         Data pipeline orchestrator.

Responsabilité unique : appeler les modules dans l'ordre et sauvegarder les résultats.
Single responsibility : call modules in order and save results.
"""
import polars as pl

from data_loader import (
    load_atcoder_problems,
    load_submissions_lazy,
    get_descriptions_dir,
    ensure_processed_dir,
    OUT_ABC_PROBLEMS,
    OUT_USER_PROFILES,
    PROCESSED_DATA_DIR,
)
from problem_parser import extract_problem_scores
from difficulty_labeler import label_abc_problems, LETTER_TO_GROUP
from user_profiling import build_user_profiles
from error_analysis import compute_error_distribution


def print_section(title: str) -> None:
    print("\n" + "=" * 52)
    print(f"  {title}")
    print("=" * 52)


def main():
    print_section("🚀 OJS Submission Analytics Pipeline")
    ensure_processed_dir()

    # ──────────────────────────────────────────────────────────────
    # PHASE 1 : Labellisation des problèmes ABC (A–F)
    # PHASE 1 : ABC problem difficulty labeling (A–F)
    # ──────────────────────────────────────────────────────────────
    print_section("PHASE 1 — Labellisation des problèmes ABC")

    df_atcoder = load_atcoder_problems()
    df_abc = label_abc_problems(df_atcoder)

    # Enrichissement avec les scores HTML — ciblé sur les IDs ABC uniquement
    abc_ids = df_abc["problem_id"].to_list()
    df_scores = extract_problem_scores(get_descriptions_dir(), abc_ids)
    df_abc = df_abc.join(df_scores, on="problem_id", how="left")

    df_abc.write_csv(OUT_ABC_PROBLEMS)
    print(f"\n  ✅ Sauvegardé : {OUT_ABC_PROBLEMS}")

    # ──────────────────────────────────────────────────────────────
    # PHASE 2 : Profilage et classification des utilisateurs
    # PHASE 2 : User profiling and classification
    # ──────────────────────────────────────────────────────────────
    print_section("PHASE 2 — Profilage & Classification des utilisateurs")

    lazy_subs = load_submissions_lazy(
        abc_ids,
        columns=["problem_id", "user_id", "status", "date"]
    )
    df_users = build_user_profiles(lazy_subs, df_abc)

    df_users.write_csv(OUT_USER_PROFILES)
    print(f"\n  ✅ Sauvegardé : {OUT_USER_PROFILES}")

    # ──────────────────────────────────────────────────────────────
    # SYNTHÈSE — Classification G1–G6 (Shimizu et al. 2025)
    # ──────────────────────────────────────────────────────────────
    print_section("SYNTHÈSE — Classification G1–G6")

    group_stats = (
        df_users
        .group_by("proficiency_group")
        .agg(pl.len().alias("n_users"))
        .sort("proficiency_group")
    )

    group_labels = {v: k for k, v in LETTER_TO_GROUP.items()}  # G1→A, G2→B...
    total = df_users.shape[0]

    # Utilisateurs sans aucun AC (jamais résolu un problème)
    no_ac = df_users.filter(pl.col("proficiency_group").is_null()).shape[0]
    classified = df_users.filter(pl.col("proficiency_group").is_not_null())

    print(f"\n  {'Groupe':<8} {'Lettre max':<14} {'Utilisateurs':>14}  {'%':>6}")
    print(f"  {'-'*48}")
    for row in group_stats.filter(pl.col("proficiency_group").is_not_null()).iter_rows(named=True):
        g = row["proficiency_group"]
        n = row["n_users"]
        letter = group_labels.get(g, "?")
        pct = n / total * 100
        print(f"  {g:<8} max {letter:<11} {n:>14,}  {pct:>5.1f}%")
    if no_ac > 0:
        print(f"  {'—':<8} {'aucun AC':<14} {no_ac:>14,}  {no_ac/total*100:>5.1f}%")
    print(f"  {'-'*48}")
    print(f"  {'TOTAL':<23} {total:>14,}  100.0%")

    # ──────────────────────────────────────────────────────────────
    # PHASE 3 : Distribution des erreurs par niveau (RQ1)
    # PHASE 3 : Error distribution by difficulty level (RQ1)
    # ──────────────────────────────────────────────────────────────
    print_section("PHASE 3 — Distribution des erreurs par niveau (RQ1)")

    lazy_subs_errors = load_submissions_lazy(
        abc_ids,
        columns=["problem_id", "status"]
    )
    df_errors = compute_error_distribution(lazy_subs_errors, df_abc)

    out_errors = PROCESSED_DATA_DIR / "atcoder_error_distribution.csv"
    df_errors.write_csv(out_errors)
    print(f"\n  ✅ Sauvegardé : {out_errors}")

    print_section("✨ Pipeline terminé avec succès !")


if __name__ == "__main__":
    main()
