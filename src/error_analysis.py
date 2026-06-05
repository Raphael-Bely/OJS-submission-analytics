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
STATUS_ORDER     = ["AC", "WA", "TLE", "RE", "CE", "Other"]
DIFFICULTY_ORDER = ["A", "B", "C", "D", "E", "F"]
GROUP_ORDER      = ["G1", "G2", "G3", "G4", "G5", "G6"]


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


# ── Resolution rate analysis ───────────────────────────────────────────────────

def _compute_user_problem_stats(
    lazy_subs: pl.LazyFrame,
    df_abc: pl.DataFrame,
) -> pl.DataFrame:
    """
    Internal helper — builds per-(user, problem) resolution metrics.

    For each (user, problem) pair, ranks submissions chronologically using the
    Unix timestamp, then determines:
      - whether the problem was ever solved (is_resolved)
      - whether it was solved on the first submission (first_try_ac)
      - how many submissions it took to reach the first AC (attempts_to_ac)

    Collects eagerly because the ranking window function cannot stay lazy
    when the result is used in two separate downstream aggregations.

    Returns:
        DataFrame with columns:
        user_id, problem_id, difficulty,
        is_resolved, first_try_ac, total_attempts,
        attempts_to_ac (null if never resolved)
    """
    lazy_labels = df_abc.lazy().select(["problem_id", "difficulty"])

    # Rank each submission by timestamp within (user, problem) — rank 1 = earliest
    # rank("ordinal") assigns unique ranks based on value: smallest timestamp → rank 1
    with_rank = (
        lazy_subs
        .select(["problem_id", "user_id", "status", "date"])
        .join(lazy_labels, on="problem_id", how="inner")
        .with_columns(
            pl.col("date")
              .rank("ordinal")
              .over(["user_id", "problem_id"])
              .cast(pl.Int32)
              .alias("submission_rank")
        )
        .collect()  # collect early — window function + used twice downstream
    )

    # Find the rank of the first AC submission per (user, problem)
    # = number of attempts needed to reach AC (inclusive)
    first_ac_rank = (
        with_rank
        .filter(pl.col("status") == "Accepted")
        .group_by(["user_id", "problem_id"])
        .agg(pl.col("submission_rank").min().alias("attempts_to_ac"))
    )

    # Aggregate to one row per (user, problem)
    user_problem = (
        with_rank
        .group_by(["user_id", "problem_id", "difficulty"])
        .agg([
            # Status of the earliest submission (submission_rank == 1)
            pl.col("status").sort_by("submission_rank").first().alias("first_status"),
            (pl.col("status") == "Accepted").any().alias("is_resolved"),
            pl.len().alias("total_attempts"),
        ])
        .with_columns(
            (pl.col("first_status") == "Accepted").alias("first_try_ac")
        )
        .drop("first_status")
        .join(first_ac_rank, on=["user_id", "problem_id"], how="left")
    )

    return user_problem


def compute_resolution_by_difficulty(
    lazy_subs: pl.LazyFrame,
    df_abc: pl.DataFrame,
) -> pl.DataFrame:
    """
    Computes problem resolution metrics per difficulty level (A–F).

    For each difficulty level, aggregates across all (user, problem) pairs:
      - resolution_rate   : % of pairs eventually resolved (any number of attempts)
      - abandon_rate      : % of pairs attempted but never resolved
      - first_try_rate    : % of pairs resolved on the very first submission
      - avg_attempts_to_ac: mean submissions before first AC (resolved pairs only)

    Args:
        lazy_subs: LazyFrame of ABC submissions
                   (required columns: problem_id, user_id, status, date)
        df_abc:    Labeled ABC problem DataFrame
                   (required columns: problem_id, difficulty)

    Returns:
        DataFrame with columns:
        difficulty | n_pairs | resolution_rate | abandon_rate |
        first_try_rate | avg_attempts_to_ac
    """
    stats = _compute_user_problem_stats(lazy_subs, df_abc)

    # avg_attempts_to_ac computed separately — only over resolved pairs
    avg_attempts = (
        stats
        .filter(pl.col("is_resolved"))
        .group_by("difficulty")
        .agg(pl.col("attempts_to_ac").mean().round(2).alias("avg_attempts_to_ac"))
    )

    result = (
        stats
        .group_by("difficulty")
        .agg([
            pl.len().alias("n_pairs"),
            pl.col("is_resolved").sum().alias("n_resolved"),
            pl.col("first_try_ac").sum().alias("n_first_try"),
        ])
        .join(avg_attempts, on="difficulty", how="left")
        .with_columns([
            (pl.col("n_resolved") / pl.col("n_pairs") * 100).round(2).alias("resolution_rate"),
            ((pl.col("n_pairs") - pl.col("n_resolved")) / pl.col("n_pairs") * 100).round(2).alias("abandon_rate"),
            (pl.col("n_first_try") / pl.col("n_pairs") * 100).round(2).alias("first_try_rate"),
        ])
        .drop(["n_resolved", "n_first_try"])
        .sort("difficulty")
    )

    # Terminal summary
    total_pairs = result["n_pairs"].sum()
    print(f"  Total (user, problem) pairs: {total_pairs:,}\n")
    print(f"  {'Level':<8} {'Resolved':>10} {'Abandoned':>11} {'First Try':>11} {'Avg attempts':>14}")
    print(f"  {'-'*56}")
    for row in result.iter_rows(named=True):
        print(
            f"  {row['difficulty']:<8}"
            f" {row['resolution_rate']:>9.1f}%"
            f" {row['abandon_rate']:>10.1f}%"
            f" {row['first_try_rate']:>10.1f}%"
            f" {row['avg_attempts_to_ac']:>13.2f}"
        )
    print()
    return result


def compute_resolution_by_group(
    lazy_subs: pl.LazyFrame,
    df_abc: pl.DataFrame,
    df_users: pl.DataFrame,
) -> pl.DataFrame:
    """
    Computes problem resolution metrics per (difficulty level × proficiency group).

    Same metrics as compute_resolution_by_difficulty, broken down by G1–G6.
    Only classified users (non-null proficiency_group) are included.

    Args:
        lazy_subs: LazyFrame of ABC submissions
                   (required columns: problem_id, user_id, status, date)
        df_abc:    Labeled ABC problem DataFrame
                   (required columns: problem_id, difficulty)
        df_users:  User profile DataFrame
                   (required columns: user_id, proficiency_group)

    Returns:
        DataFrame with columns:
        difficulty | proficiency_group | n_pairs | resolution_rate |
        abandon_rate | first_try_rate | avg_attempts_to_ac
    """
    stats = _compute_user_problem_stats(lazy_subs, df_abc)

    # Join with proficiency groups — exclude unclassified users
    lazy_groups = (
        df_users.lazy()
        .select(["user_id", "proficiency_group"])
        .filter(pl.col("proficiency_group").is_not_null())
        .collect()
    )
    stats_with_group = stats.join(lazy_groups, on="user_id", how="inner")

    # avg_attempts_to_ac over resolved pairs only
    avg_attempts = (
        stats_with_group
        .filter(pl.col("is_resolved"))
        .group_by(["difficulty", "proficiency_group"])
        .agg(pl.col("attempts_to_ac").mean().round(2).alias("avg_attempts_to_ac"))
    )

    result = (
        stats_with_group
        .group_by(["difficulty", "proficiency_group"])
        .agg([
            pl.len().alias("n_pairs"),
            pl.col("is_resolved").sum().alias("n_resolved"),
            pl.col("first_try_ac").sum().alias("n_first_try"),
        ])
        .join(avg_attempts, on=["difficulty", "proficiency_group"], how="left")
        .with_columns([
            (pl.col("n_resolved") / pl.col("n_pairs") * 100).round(2).alias("resolution_rate"),
            ((pl.col("n_pairs") - pl.col("n_resolved")) / pl.col("n_pairs") * 100).round(2).alias("abandon_rate"),
            (pl.col("n_first_try") / pl.col("n_pairs") * 100).round(2).alias("first_try_rate"),
        ])
        .drop(["n_resolved", "n_first_try"])
        .sort(["difficulty", "proficiency_group"])
    )

    # Terminal summary — resolution rate on level A by group
    total_pairs = result["n_pairs"].sum()
    print(f"  Total (user, problem) pairs: {total_pairs:,}\n")
    print(f"  Resolution rate on level A by group:")
    for g in GROUP_ORDER:
        subset = result.filter(
            (pl.col("difficulty") == "A") & (pl.col("proficiency_group") == g)
        )
        if not subset.is_empty():
            print(f"    {g}  resolved: {subset['resolution_rate'].item():>5.1f}%  first try: {subset['first_try_rate'].item():>5.1f}%")
    print()
    return result
