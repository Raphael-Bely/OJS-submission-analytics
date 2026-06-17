"""
main.py — Data pipeline orchestrator.

Single responsibility: call modules in order and save results.
No processing logic — no file paths (delegated to data_loader).
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
from error_analysis import (
    compute_error_distribution,
    compute_error_by_group_and_difficulty,
    compute_language_distribution,
    compute_language_distribution_by_group,
    compute_error_by_language,
    compute_resolution_by_difficulty,
    compute_resolution_by_group,
    compute_resolution_by_first_error,
    compute_resolution_by_first_error_and_group,
    compute_error_sequences,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 52)
    print(f"  {title}")
    print("=" * 52)


def main():
    print_section("OJS Submission Analytics Pipeline")
    ensure_processed_dir()

    # ── PHASE 1: ABC problem difficulty labeling (A–F) ────────────────────────
    print_section("PHASE 1 — ABC Problem Labeling")

    df_atcoder = load_atcoder_problems()
    df_abc = label_abc_problems(df_atcoder)

    # Enrich with HTML scores — targeted to ABC IDs only
    abc_ids = df_abc["problem_id"].to_list()
    df_scores = extract_problem_scores(get_descriptions_dir(), abc_ids)
    df_abc = df_abc.join(df_scores, on="problem_id", how="left")

    df_abc.write_csv(OUT_ABC_PROBLEMS)
    print(f"\n  Saved: {OUT_ABC_PROBLEMS}")

    # ── PHASE 2: User profiling and G1–G6 classification ─────────────────────
    print_section("PHASE 2 — User Profiling & Classification")

    lazy_subs = load_submissions_lazy(
        abc_ids,
        columns=["problem_id", "user_id", "status", "date"]
    )
    df_users = build_user_profiles(lazy_subs, df_abc)

    df_users.write_csv(OUT_USER_PROFILES)
    print(f"\n  Saved: {OUT_USER_PROFILES}")

    # ── PHASE 2 SUMMARY: G1–G6 breakdown ─────────────────────────────────────
    print_section("SUMMARY — G1–G6 Classification")

    group_stats = (
        df_users
        .group_by("proficiency_group")
        .agg(pl.len().alias("n_users"))
        .sort("proficiency_group")
    )

    group_labels = {v: k for k, v in LETTER_TO_GROUP.items()}
    total = df_users.shape[0]
    no_ac = df_users.filter(pl.col("proficiency_group").is_null()).shape[0]

    print(f"\n  {'Group':<8} {'Max level':<14} {'Users':>14}  {'%':>6}")
    print(f"  {'-'*48}")
    for row in group_stats.filter(pl.col("proficiency_group").is_not_null()).iter_rows(named=True):
        g = row["proficiency_group"]
        n = row["n_users"]
        letter = group_labels.get(g, "?")
        pct = n / total * 100
        print(f"  {g:<8} max {letter:<11} {n:>14,}  {pct:>5.1f}%")
    if no_ac > 0:
        print(f"  {'—':<8} {'no AC':<14} {no_ac:>14,}  {no_ac/total*100:>5.1f}%")
    print(f"  {'-'*48}")
    print(f"  {'TOTAL':<23} {total:>14,}  100.0%")

    # ── PHASE 3a: Error distribution by difficulty level (RQ1 — all users) ───
    print_section("PHASE 3a — Error Distribution by Difficulty Level (RQ1)")

    lazy_subs_errors = load_submissions_lazy(
        abc_ids,
        columns=["problem_id", "status"]
    )
    df_errors = compute_error_distribution(lazy_subs_errors, df_abc)

    out_errors = PROCESSED_DATA_DIR / "atcoder_error_distribution.csv"
    df_errors.write_csv(out_errors)
    print(f"\n  Saved: {out_errors}")

    # ── PHASE 3b: Error distribution by difficulty x group (RQ1 — Table 4) ───
    print_section("PHASE 3b — Error Distribution by Difficulty x Group (Table 4)")

    lazy_subs_grouped = load_submissions_lazy(
        abc_ids,
        columns=["problem_id", "user_id", "status"]
    )
    df_errors_by_group = compute_error_by_group_and_difficulty(
        lazy_subs_grouped, df_abc, df_users
    )

    out_errors_by_group = PROCESSED_DATA_DIR / "atcoder_error_by_group_difficulty.csv"
    df_errors_by_group.write_csv(out_errors_by_group)
    print(f"\n  Saved: {out_errors_by_group}")

    # ── PHASE 4a: Language distribution by difficulty ─────────────────────────
    print_section("PHASE 4a — Language Distribution by Difficulty")

    lazy_lang = load_submissions_lazy(abc_ids, columns=["problem_id", "filename_ext"])
    df_lang = compute_language_distribution(lazy_lang, df_abc)
    df_lang.write_csv(PROCESSED_DATA_DIR / "atcoder_language_distribution.csv")
    print(f"\n  Saved: {PROCESSED_DATA_DIR / 'atcoder_language_distribution.csv'}")

    # ── PHASE 4b: Language distribution by difficulty x group ─────────────────
    print_section("PHASE 4b — Language Distribution by Difficulty x Group")

    lazy_lang_group = load_submissions_lazy(abc_ids, columns=["problem_id", "user_id", "filename_ext"])
    df_lang_group = compute_language_distribution_by_group(lazy_lang_group, df_abc, df_users)
    df_lang_group.write_csv(PROCESSED_DATA_DIR / "atcoder_language_by_group.csv")
    print(f"\n  Saved: {PROCESSED_DATA_DIR / 'atcoder_language_by_group.csv'}")

    # ── PHASE 4c: Error distribution by language x difficulty ─────────────────
    print_section("PHASE 4c — Error Distribution by Language x Difficulty")

    lazy_lang_err = load_submissions_lazy(abc_ids, columns=["problem_id", "filename_ext", "status"])
    df_lang_err = compute_error_by_language(lazy_lang_err, df_abc)
    df_lang_err.write_csv(PROCESSED_DATA_DIR / "atcoder_error_by_language.csv")
    print(f"\n  Saved: {PROCESSED_DATA_DIR / 'atcoder_error_by_language.csv'}")

    # ── PHASE 5a: Resolution rate by difficulty ───────────────────────────────
    print_section("PHASE 5a — Resolution Rate by Difficulty")

    lazy_resolution = load_submissions_lazy(
        abc_ids,
        columns=["problem_id", "user_id", "status", "date"]
    )
    df_resolution = compute_resolution_by_difficulty(lazy_resolution, df_abc)
    df_resolution.write_csv(PROCESSED_DATA_DIR / "atcoder_resolution_by_difficulty.csv")
    print(f"\n  Saved: {PROCESSED_DATA_DIR / 'atcoder_resolution_by_difficulty.csv'}")

    # ── PHASE 5b: Resolution rate by difficulty x group ───────────────────────
    print_section("PHASE 5b — Resolution Rate by Difficulty x Group")

    lazy_resolution_group = load_submissions_lazy(
        abc_ids,
        columns=["problem_id", "user_id", "status", "date"]
    )
    df_resolution_group = compute_resolution_by_group(lazy_resolution_group, df_abc, df_users)
    df_resolution_group.write_csv(PROCESSED_DATA_DIR / "atcoder_resolution_by_group.csv")
    print(f"\n  Saved: {PROCESSED_DATA_DIR / 'atcoder_resolution_by_group.csv'}")

    # ── PHASE 6a: Resolution rate by first error type x difficulty ────────────
    print_section("PHASE 6a — Resolution Rate by First Error Type x Difficulty")

    lazy_first_error = load_submissions_lazy(
        abc_ids,
        columns=["problem_id", "user_id", "status", "date"]
    )
    df_first_error = compute_resolution_by_first_error(lazy_first_error, df_abc)
    df_first_error.write_csv(PROCESSED_DATA_DIR / "atcoder_resolution_by_first_error.csv")
    print(f"\n  Saved: {PROCESSED_DATA_DIR / 'atcoder_resolution_by_first_error.csv'}")

    # ── PHASE 6b: Resolution rate by first error type x difficulty x group ───
    print_section("PHASE 6b — Resolution Rate by First Error Type x Difficulty x Group")

    lazy_first_error_group = load_submissions_lazy(
        abc_ids,
        columns=["problem_id", "user_id", "status", "date"]
    )
    df_first_error_group = compute_resolution_by_first_error_and_group(
        lazy_first_error_group, df_abc, df_users
    )
    df_first_error_group.write_csv(PROCESSED_DATA_DIR / "atcoder_resolution_by_first_error_group.csv")
    print(f"\n  Saved: {PROCESSED_DATA_DIR / 'atcoder_resolution_by_first_error_group.csv'}")
    
    # ── PHASE 7: Error sequence paths × difficulty × group ────────────────────
    print_section("PHASE 7 — Error Sequence Paths × Difficulty × Group")

    lazy_sequences = load_submissions_lazy(
        abc_ids,
        columns=["problem_id", "user_id", "status", "date"]
    )
    df_sequences = compute_error_sequences(lazy_sequences, df_abc, df_users)
    df_sequences.write_csv(PROCESSED_DATA_DIR / "atcoder_error_sequences.csv")
    print(f"\n  Saved: {PROCESSED_DATA_DIR / 'atcoder_error_sequences.csv'}")

    print_section("Pipeline completed successfully")


if __name__ == "__main__":
    main()
