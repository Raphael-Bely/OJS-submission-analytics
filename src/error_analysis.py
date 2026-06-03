"""
error_analysis.py — Error distribution analysis by difficulty level and language.

Methodology: Shimizu et al. (2025) — RQ1 + language extension.
Analysis unit: the submission (not the user).
"""
import polars as pl


# Normalization: full status names → short codes
# Values observed in CodeNet AtCoder data.
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

# Language normalization: filename_ext → canonical language name.
# Extensions verified to exist in the AtCoder ABC subset of CodeNet.
# All other extensions (js, kt, hs, php, scala, nim, ...) → "Other".
EXT_TO_LANGUAGE = {
    "cpp":  "C++",
    "py":   "Python",
    "java": "Java",
    "c":    "C",
    "rb":   "Ruby",
    "cs":   "C#",
    "rs":   "Rust",
    "go":   "Go",
}
# All other extensions (js, kt, hs, php, d, scala, nim, sh, ...) → "Other"

LANGUAGE_ORDER = ["C++", "Python", "Java", "C", "Ruby", "C#", "Rust", "Go", "Other"]

# Display order for charts and terminal output
STATUS_ORDER = ["AC", "WA", "TLE", "RE", "CE", "Other"]
DIFFICULTY_ORDER = ["A", "B", "C", "D", "E", "F"]


def compute_error_distribution(
    lazy_subs: pl.LazyFrame,
    df_abc: pl.DataFrame,
) -> pl.DataFrame:
    """
    Computes submission status distribution per difficulty letter.

    Args:
        lazy_subs: LazyFrame of ABC submissions
                   (required columns: problem_id, status)
        df_abc:    Labeled ABC problem DataFrame
                   (required columns: problem_id, difficulty)

    Returns:
        DataFrame with columns:
        difficulty | status_code | n_submissions | pct
        Sorted by difficulty then status_code.
    """
    lazy_labels = df_abc.lazy().select(["problem_id", "difficulty"])

    # Join + normalize status names to short codes
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

    # Total submissions per difficulty level (denominator for percentages)
    totals = (
        joined
        .group_by("difficulty")
        .agg(pl.len().alias("total_submissions"))
    )

    # Count per (difficulty, status_code)
    counts = (
        joined
        .group_by(["difficulty", "status_code"])
        .agg(pl.len().alias("n_submissions"))
    )

    # Join + compute percentages
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

    # Terminal summary
    total_subs = result["n_submissions"].sum()
    print(f"  Total ABC submissions analyzed: {total_subs:,}\n")

    main_statuses = ["AC", "WA", "TLE", "CE", "RE"]
    header = f"  {'Level':<8}" + "".join(f"{s:>8}%" for s in main_statuses)
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


def compute_error_by_group_and_difficulty(
    lazy_subs: pl.LazyFrame,
    df_abc: pl.DataFrame,
    df_users: pl.DataFrame,
) -> pl.DataFrame:
    """
    Computes submission status distribution per (difficulty level x proficiency group).
    Replicates Table 4 of Shimizu et al. (2025).

    Only classified users (G1–G6) are included. Users with no AC (null group)
    are excluded — they cannot be meaningfully assigned to a proficiency level.

    Args:
        lazy_subs: LazyFrame of ABC submissions
                   (required columns: problem_id, user_id, status)
        df_abc:    Labeled ABC problem DataFrame
                   (required columns: problem_id, difficulty)
        df_users:  User profile DataFrame
                   (required columns: user_id, proficiency_group)

    Returns:
        DataFrame with columns:
        difficulty | proficiency_group | status_code | n_submissions | pct
        Sorted by difficulty, proficiency_group, status_code.
    """
    lazy_labels = df_abc.lazy().select(["problem_id", "difficulty"])

    # Only keep classified users (exclude null proficiency_group)
    lazy_groups = (
        df_users.lazy()
        .select(["user_id", "proficiency_group"])
        .filter(pl.col("proficiency_group").is_not_null())
    )

    # Three-way join: submissions × difficulty labels × user groups
    joined = (
        lazy_subs
        .select(["problem_id", "user_id", "status"])
        .join(lazy_labels, on="problem_id", how="inner")
        .join(lazy_groups, on="user_id", how="inner")
        .with_columns(
            pl.col("status")
              .replace(STATUS_MAP, default="Other")
              .alias("status_code")
        )
    )

    # Total submissions per (difficulty, group) — denominator for percentages
    totals = (
        joined
        .group_by(["difficulty", "proficiency_group"])
        .agg(pl.len().alias("total_submissions"))
    )

    # Count per (difficulty, group, status_code)
    counts = (
        joined
        .group_by(["difficulty", "proficiency_group", "status_code"])
        .agg(pl.len().alias("n_submissions"))
    )

    # Join + compute percentages
    result = (
        counts
        .join(totals, on=["difficulty", "proficiency_group"], how="left")
        .with_columns(
            (pl.col("n_submissions") / pl.col("total_submissions") * 100)
              .round(2)
              .alias("pct")
        )
        .drop("total_submissions")
        .sort(["difficulty", "proficiency_group", "status_code"])
        .collect()
    )

    # Terminal summary — AC rate per (difficulty, group)
    total_subs = result["n_submissions"].sum()
    print(f"  Total submissions analyzed: {total_subs:,}\n")

    groups = sorted(result["proficiency_group"].unique().to_list())
    header = f"  {'Level':<6}" + "".join(f"{g:>9}" for g in groups)
    print(f"  AC rate per group (%):")
    print(header)
    print("  " + "-" * (6 + 9 * len(groups)))

    for diff in DIFFICULTY_ORDER:
        row_parts = []
        for g in groups:
            subset = result.filter(
                (pl.col("difficulty") == diff) &
                (pl.col("proficiency_group") == g) &
                (pl.col("status_code") == "AC")
            )
            val = subset["pct"].item() if not subset.is_empty() else None
            row_parts.append(f"{val:>8.1f}" if val is not None else f"{'—':>9}")
        print(f"  {diff:<6}" + "".join(row_parts))

    print()
    return result


# ── Shared helper ──────────────────────────────────────────────────────────────

def _normalize_language(lazy: pl.LazyFrame) -> pl.LazyFrame:
    """Adds a 'language' column by mapping filename_ext → canonical name."""
    return lazy.with_columns(
        pl.col("filename_ext")
          .replace(EXT_TO_LANGUAGE, default="Other")
          .alias("language")
    )


# ── Language analysis functions ────────────────────────────────────────────────

def compute_language_distribution(
    lazy_subs: pl.LazyFrame,
    df_abc: pl.DataFrame,
) -> pl.DataFrame:
    """
    Computes submission count per language per difficulty level.

    Args:
        lazy_subs: LazyFrame of ABC submissions
                   (required columns: problem_id, filename_ext)
        df_abc:    Labeled ABC problem DataFrame
                   (required columns: problem_id, difficulty)

    Returns:
        DataFrame with columns: difficulty | language | n_submissions | pct
    """
    lazy_labels = df_abc.lazy().select(["problem_id", "difficulty"])

    joined = (
        lazy_subs
        .select(["problem_id", "filename_ext"])
        .join(lazy_labels, on="problem_id", how="inner")
        .pipe(_normalize_language)
    )

    totals = joined.group_by("difficulty").agg(pl.len().alias("total"))
    counts = joined.group_by(["difficulty", "language"]).agg(pl.len().alias("n_submissions"))

    result = (
        counts
        .join(totals, on="difficulty", how="left")
        .with_columns(
            (pl.col("n_submissions") / pl.col("total") * 100).round(2).alias("pct")
        )
        .drop("total")
        .sort(["difficulty", "n_submissions"], descending=[False, True])
        .collect()
    )

    total_subs = result["n_submissions"].sum()
    print(f"  Total submissions: {total_subs:,}")
    print(f"  Languages found:   {sorted(result['language'].unique().to_list())}")
    return result


def compute_language_distribution_by_group(
    lazy_subs: pl.LazyFrame,
    df_abc: pl.DataFrame,
    df_users: pl.DataFrame,
) -> pl.DataFrame:
    """
    Computes submission count per (difficulty, proficiency group, language).

    Args:
        lazy_subs: LazyFrame of ABC submissions
                   (required columns: problem_id, user_id, filename_ext)
        df_abc:    Labeled ABC problem DataFrame
        df_users:  User profile DataFrame
                   (required columns: user_id, proficiency_group)

    Returns:
        DataFrame with columns:
        difficulty | proficiency_group | language | n_submissions | pct
    """
    lazy_labels = df_abc.lazy().select(["problem_id", "difficulty"])
    lazy_groups = (
        df_users.lazy()
        .select(["user_id", "proficiency_group"])
        .filter(pl.col("proficiency_group").is_not_null())
    )

    joined = (
        lazy_subs
        .select(["problem_id", "user_id", "filename_ext"])
        .join(lazy_labels, on="problem_id", how="inner")
        .join(lazy_groups, on="user_id", how="inner")
        .pipe(_normalize_language)
    )

    totals = (
        joined
        .group_by(["difficulty", "proficiency_group"])
        .agg(pl.len().alias("total"))
    )
    counts = (
        joined
        .group_by(["difficulty", "proficiency_group", "language"])
        .agg(pl.len().alias("n_submissions"))
    )

    result = (
        counts
        .join(totals, on=["difficulty", "proficiency_group"], how="left")
        .with_columns(
            (pl.col("n_submissions") / pl.col("total") * 100).round(2).alias("pct")
        )
        .drop("total")
        .sort(["difficulty", "proficiency_group", "n_submissions"], descending=[False, False, True])
        .collect()
    )

    print(f"  Total submissions: {result['n_submissions'].sum():,}")
    return result


def compute_error_by_language(
    lazy_subs: pl.LazyFrame,
    df_abc: pl.DataFrame,
) -> pl.DataFrame:
    """
    Computes error type distribution per (language, difficulty level).

    Args:
        lazy_subs: LazyFrame of ABC submissions
                   (required columns: problem_id, filename_ext, status)
        df_abc:    Labeled ABC problem DataFrame
                   (required columns: problem_id, difficulty)

    Returns:
        DataFrame with columns:
        difficulty | language | status_code | n_submissions | pct
    """
    lazy_labels = df_abc.lazy().select(["problem_id", "difficulty"])

    joined = (
        lazy_subs
        .select(["problem_id", "filename_ext", "status"])
        .join(lazy_labels, on="problem_id", how="inner")
        .pipe(_normalize_language)
        .with_columns(
            pl.col("status")
              .replace(STATUS_MAP, default="Other")
              .alias("status_code")
        )
    )

    totals = (
        joined
        .group_by(["difficulty", "language"])
        .agg(pl.len().alias("total"))
    )
    counts = (
        joined
        .group_by(["difficulty", "language", "status_code"])
        .agg(pl.len().alias("n_submissions"))
    )

    result = (
        counts
        .join(totals, on=["difficulty", "language"], how="left")
        .with_columns(
            (pl.col("n_submissions") / pl.col("total") * 100).round(2).alias("pct")
        )
        .drop("total")
        .sort(["difficulty", "language", "status_code"])
        .collect()
    )

    # Terminal summary: AC rate per language on level A
    print(f"  Total submissions analyzed: {result['n_submissions'].sum():,}\n")
    print(f"  AC rate on level A by language:")
    for lang in LANGUAGE_ORDER:
        subset = result.filter(
            (pl.col("difficulty") == "A") &
            (pl.col("language") == lang) &
            (pl.col("status_code") == "AC")
        )
        if not subset.is_empty():
            print(f"    {lang:<10} {subset['pct'].item():>5.1f}%")
    print()
    return result
