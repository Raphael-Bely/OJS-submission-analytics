"""
error_analysis.py — Analyse de la distribution des erreurs par niveau de difficulté.
                    Error distribution analysis by difficulty level.

Méthodologie : Shimizu et al. (2025) — RQ1.
Unité d'analyse : la soumission (pas l'utilisateur).
Pour chaque niveau de difficulté A–F, calcule la proportion de chaque
type de statut (AC, WA, CE, TLE, RE, Other) parmi toutes les soumissions.

Analysis unit: the submission (not the user).
For each difficulty level A–F, computes the proportion of each
status type (AC, WA, CE, TLE, RE, Other) across all submissions.
"""
import polars as pl


# ── Normalisation des statuts → codes courts ──────────────────────────────────
# Valeurs observées dans les données CodeNet AtCoder.
STATUS_MAP = {
    "Accepted":              "AC",
    "Wrong Answer":          "WA",
    "Compile Error":         "CE",
    "Time Limit Exceeded":   "TLE",
    "Runtime Error":         "RE",
    "Memory Limit Exceeded": "Other",
    "Output Limit Exceeded": "Other",
    "Judge Not Available":   "Other",
}

# Ordre d'affichage pour les graphiques
STATUS_ORDER = ["AC", "WA", "TLE", "RE", "CE", "Other"]
DIFFICULTY_ORDER = ["A", "B", "C", "D", "E", "F"]


def compute_error_distribution(
    lazy_subs: pl.LazyFrame,
    df_abc: pl.DataFrame,
) -> pl.DataFrame:
    """
    Calcule la distribution des statuts de soumission par lettre de difficulté.

    Computes submission status distribution per difficulty letter.

    Args:
        lazy_subs : LazyFrame des soumissions ABC
                    (colonnes requises : problem_id, status)
        df_abc    : DataFrame des problèmes ABC labellisés
                    (colonnes requises : problem_id, difficulty)

    Returns:
        DataFrame avec colonnes :
        difficulty | status_code | n_submissions | pct
        Trié par difficulty puis status_code.
    """
    lazy_labels = df_abc.lazy().select(["problem_id", "difficulty"])

    # Jointure + normalisation des statuts
    joined = (
        lazy_subs
        .select(["problem_id", "status"])
        .join(lazy_labels, on="problem_id", how="inner")
        .with_columns(
            pl.col("status")
              .replace(STATUS_MAP, default="Other")
              .alias("status_code")
        )
    )

    # Total de soumissions par difficulté (pour le calcul des %)
    totals = (
        joined
        .group_by("difficulty")
        .agg(pl.len().alias("total_submissions"))
    )

    # Comptage par (difficulty, status_code)
    counts = (
        joined
        .group_by(["difficulty", "status_code"])
        .agg(pl.len().alias("n_submissions"))
    )

    # Jointure + calcul des pourcentages
    result = (
        counts
        .join(totals, on="difficulty", how="left")
        .with_columns(
            (pl.col("n_submissions") / pl.col("total_submissions") * 100)
              .round(2)
              .alias("pct")
        )
        .drop("total_submissions")
        .sort(["difficulty", "status_code"])
        .collect()
    )

    # ── Résumé console ────────────────────────────────────────────────────────
    total_subs = result["n_submissions"].sum()
    print(f"  Total soumissions ABC analysées : {total_subs:,}\n")

    # Tableau pivot : difficulty × status_code → pct
    # On pivote uniquement les statuts principaux pour un affichage lisible
    main_statuses = ["AC", "WA", "TLE", "CE", "RE"]
    header = f"  {'Niveau':<8}" + "".join(f"{s:>8}%" for s in main_statuses)
    print(header)
    print("  " + "-" * (8 + 9 * len(main_statuses)))

    for diff in DIFFICULTY_ORDER:
        row_data = result.filter(pl.col("difficulty") == diff)
        if row_data.is_empty():
            continue
        pct_map = dict(zip(row_data["status_code"].to_list(), row_data["pct"].to_list()))
        row_str = f"  {diff:<8}" + "".join(f"{pct_map.get(s, 0.0):>8.1f}" for s in main_statuses)
        print(row_str)

    print()
    return result
