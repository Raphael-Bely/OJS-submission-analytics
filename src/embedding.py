"""
embedding.py — Code embedding pipeline for RQ2.

Samples submissions stratified by (difficulty × status_code), reads source code
from Project CodeNet, embeds with CodeBERT or TF-IDF (baseline).

Usage:
    python src/embedding.py tfidf       # fast baseline, no GPU needed
    python src/embedding.py codebert    # slower, requires: pip install torch transformers

Outputs in data/processed/embeddings/:
    embeddings_{method}.npy   — float32 array, shape (N, D)
    metadata_{method}.csv     — N rows with submission_id, difficulty, status_code, etc.
"""
from pathlib import Path
import random
import sys

import numpy as np
import polars as pl

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_DIR   = Path("./data/Project_CodeNet")
PROCESSED_DIR  = Path("./data/processed")
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
METADATA_DIR   = RAW_DATA_DIR / "metadata"

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLES_PER_CELL    = 500
TARGET_DIFFICULTIES = ["B", "C", "D", "E"]
TARGET_STATUSES     = ["AC", "WA", "TLE", "CE", "RE"]
MAX_TOKENS          = 512

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


# ── Step 1 — Sampling ─────────────────────────────────────────────────────────

def _load_eligible(
    abc_ids: list[str],
    df_abc: pl.DataFrame,
    df_users: pl.DataFrame,
    language: str | None = None,
    problem_id: str | None = None,
) -> pl.DataFrame:
    """
    Loads all ABC submissions with (submission_id, problem_id, user_id, language,
    filename_ext, status_code, difficulty, proficiency_group).
    Filtered to TARGET_DIFFICULTIES × TARGET_STATUSES.
    If language is given (e.g. "C++"), keeps only submissions whose language
    starts with that prefix (catches C++14, C++17, C++20, etc.).
    If problem_id is given, restricts to that single problem only.
    """
    lazy_frames = []
    for pid in abc_ids:
        fp = METADATA_DIR / f"{pid}.csv"
        if fp.exists() and fp.stat().st_size > 50:
            try:
                lf = pl.scan_csv(str(fp), schema_overrides={
                    "submission_id": pl.Utf8,
                    "problem_id":    pl.Utf8,
                    "user_id":       pl.Utf8,
                    "status":        pl.Utf8,
                    "language":      pl.Utf8,
                    "filename_ext":  pl.Utf8,
                })
                lf = lf.select([
                    "submission_id", "problem_id", "user_id",
                    "language", "filename_ext", "status",
                ])
                lazy_frames.append(lf)
            except Exception:
                pass

    if not lazy_frames:
        raise ValueError(f"No submission metadata found in {METADATA_DIR}")

    lazy_labels = df_abc.lazy().select(["problem_id", "difficulty"])
    lazy_groups = (
        df_users.lazy()
        .select(["user_id", "proficiency_group"])
        .filter(pl.col("proficiency_group").is_not_null())
    )

    lazy = (
        pl.concat(lazy_frames)
        .join(lazy_labels, on="problem_id", how="inner")
        .join(lazy_groups, on="user_id", how="inner")
        .with_columns(
            pl.col("status").replace(STATUS_MAP).fill_null("Other").alias("status_code")
        )
        .filter(pl.col("status_code").is_in(TARGET_STATUSES))
    )
    if not problem_id:
        lazy = lazy.filter(pl.col("difficulty").is_in(TARGET_DIFFICULTIES))
    if language:
        lazy = lazy.filter(pl.col("language").str.starts_with(language))
    if problem_id:
        lazy = lazy.filter(pl.col("problem_id") == problem_id)
    return (
        lazy
        .select([
            "submission_id", "problem_id", "user_id", "difficulty",
            "proficiency_group", "status_code", "language", "filename_ext",
        ])
        .collect()
    )


def sample_submissions(
    abc_ids: list[str],
    df_abc: pl.DataFrame,
    df_users: pl.DataFrame,
    n_per_cell: int = SAMPLES_PER_CELL,
    seed: int = 42,
    language: str | None = None,
    problem_id: str | None = None,
) -> pl.DataFrame:
    """
    Stratified sample: up to n_per_cell rows per (difficulty × status_code) cell.
    If problem_id is given, samples per status_code only (difficulty is fixed).
    """
    lang_str = f" (langage : {language}*)" if language else ""
    pid_str  = f" (problème : {problem_id})" if problem_id else ""
    print(f"Loading submission metadata for sampling{lang_str}{pid_str}...")
    df_all = _load_eligible(abc_ids, df_abc, df_users, language=language, problem_id=problem_id)
    print(f"  Eligible submissions: {df_all.height:,}")

    rng = random.Random(seed)
    sampled = []

    if problem_id:
        # Un seul problème → une seule difficulté, stratifier par status uniquement
        for status in TARGET_STATUSES:
            cell = df_all.filter(pl.col("status_code") == status)
            n = min(n_per_cell, cell.height)
            if n == 0:
                print(f"  {status}: 0 — skipped")
                continue
            idx = sorted(rng.sample(range(cell.height), n))
            sampled.append(cell[idx])
            print(f"  {status}: {n:,}")
    else:
        for diff in TARGET_DIFFICULTIES:
            for status in TARGET_STATUSES:
                cell = df_all.filter(
                    (pl.col("difficulty") == diff) & (pl.col("status_code") == status)
                )
                n = min(n_per_cell, cell.height)
                if n == 0:
                    print(f"  {diff} × {status}: 0 — skipped")
                    continue
                idx = sorted(rng.sample(range(cell.height), n))
                sampled.append(cell[idx])
                print(f"  {diff} × {status}: {n:,}")

    if not sampled:
        raise RuntimeError(
            f"0 soumissions éligibles pour problem_id='{problem_id}'. "
            f"Vérifiez l'ID avec : python src/embedding.py list"
        )
    result = pl.concat(sampled)
    print(f"\n  Total sampled: {result.height:,} submissions")
    return result


# ── Step 2 — Read source code ─────────────────────────────────────────────────

def read_source_code(problem_id: str, language: str, submission_id: str, ext: str) -> str | None:
    """
    Reads from: data/Project_CodeNet/data/{problem_id}/{language}/{submission_id}.{ext}
    Returns None if missing or unreadable.
    """
    path = RAW_DATA_DIR / "data" / problem_id / language / f"{submission_id}.{ext}"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def load_codes(df_sample: pl.DataFrame) -> tuple[list[str], list[bool]]:
    """
    Reads source code for all sampled submissions.
    Returns (codes, valid_mask) where valid_mask[i] = False means file was missing.
    """
    print("Reading source code files...")
    codes, valid_mask = [], []
    for row in df_sample.iter_rows(named=True):
        code = read_source_code(
            row["problem_id"], row["language"], row["submission_id"], row["filename_ext"]
        )
        if code and len(code.strip()) > 10:
            codes.append(code)
            valid_mask.append(True)
        else:
            valid_mask.append(False)
    n_missing = valid_mask.count(False)
    print(f"  Read: {len(codes):,}  Missing/empty: {n_missing:,}")
    return codes, valid_mask


# ── Step 3a — TF-IDF baseline (fast, no GPU) ─────────────────────────────────

def embed_tfidf(codes: list[str]) -> np.ndarray:
    """
    TF-IDF over identifier tokens. Fast baseline, no semantic understanding.
    Returns float32 array of shape (N, 10000).

    Requires: pip install scikit-learn
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("Computing TF-IDF embeddings...")
    vec = TfidfVectorizer(
        token_pattern=r"[a-zA-Z_]\w*",  # identifier-like tokens only
        max_features=10_000,
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(codes)
    print(f"  Vocabulary size: {len(vec.vocabulary_):,}")
    return matrix.toarray().astype(np.float32)


# ── Step 3b — CodeBERT (slow, requires torch + transformers) ─────────────────

def embed_codebert(
    codes: list[str],
    batch_size: int = 16,
    device: str = "cpu",
) -> np.ndarray:
    """
    Embeds source code with microsoft/codebert-base (CLS token, 768 dims).
    Returns float32 array of shape (N, 768).

    Requires: pip install transformers torch
    Speed: ~1 submission/sec on CPU, ~50-100/sec on GPU.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading microsoft/codebert-base on {device}...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model     = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
    model.eval()

    embeddings = []
    for i in range(0, len(codes), batch_size):
        batch   = codes[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            out = model(**encoded)

        cls_vecs = out.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_vecs)

        done = min(i + batch_size, len(codes))
        if done % (batch_size * 10) == 0 or done == len(codes):
            print(f"  {done}/{len(codes)}")

    return np.vstack(embeddings).astype(np.float32)


# ── Step 4 — Save ─────────────────────────────────────────────────────────────

def save(embeddings: np.ndarray, df_meta: pl.DataFrame, method: str) -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_emb  = EMBEDDINGS_DIR / f"embeddings_{method}.npy"
    out_meta = EMBEDDINGS_DIR / f"metadata_{method}.csv"
    np.save(out_emb, embeddings)
    df_meta.write_csv(out_meta)
    print(f"\n  embeddings : {out_emb}  shape={embeddings.shape}")
    print(f"  metadata   : {out_meta}  rows={df_meta.height:,}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    abc_ids: list[str],
    df_abc: pl.DataFrame,
    df_users: pl.DataFrame,
    method: str = "tfidf",
    n_per_cell: int = SAMPLES_PER_CELL,
    device: str = "cpu",
    batch_size: int = 16,
    seed: int = 42,
    language: str | None = None,
    problem_id: str | None = None,
) -> None:
    df_sample = sample_submissions(
        abc_ids, df_abc, df_users, n_per_cell, seed,
        language=language, problem_id=problem_id,
    )

    codes, valid_mask = load_codes(df_sample)
    df_valid = df_sample.filter(pl.Series("valid", valid_mask))

    if len(codes) == 0:
        raise RuntimeError("No source code files found. Check CodeNet data path.")

    lang_tag = "_" + language.lower().replace("+", "p") if language else ""
    pid_tag  = "_" + problem_id.replace("/", "-") if problem_id else ""
    method_tag = f"{method}{lang_tag}{pid_tag}"

    print(f"\nEmbedding with {method_tag}...")
    if method == "tfidf":
        emb = embed_tfidf(codes)
    elif method == "codebert":
        emb = embed_codebert(codes, batch_size=batch_size, device=device)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'tfidf' or 'codebert'.")

    save(emb, df_valid, method_tag)


# ── CLI entry point ───────────────────────────────────────────────────────────

HELP_TEXT = """\
Usage : python src/embedding.py <méthode> [device] [langage] [problem_id]

Méthodes :
  tfidf        Baseline lexicale (TF-IDF sur tokens identifiants) — rapide, CPU
  codebert     Encodeur sémantique microsoft/codebert-base (768 dims) — GPU conseillé
  list [D]     Liste les problem_id valides (top 30 par volume de soumissions),
               filtrable par difficulté : python src/embedding.py list D
  help         Affiche cette aide

Arguments positionnels :
  device       cpu (défaut) | mps (Apple Silicon) | cuda (NVIDIA) — ignoré par tfidf
  langage      Préfixe de langage : "C++" (couvre C++14/17/20…), "Python", "Java"…
               "" (chaîne vide) = tous les langages
  problem_id   Restreint à un seul problème (ex. p02616).
               Sans problem_id : échantillon stratifié difficulté (B–E) × verdict,
               500 soumissions max par case.
               Avec problem_id : stratifié par verdict uniquement.

Sorties (data/processed/embeddings/) :
  embeddings_{méthode}[_{langage}][_{problème}].npy   matrice (N, D) float32
  metadata_{méthode}[_{langage}][_{problème}].csv     métadonnées alignées
  Les variantes ne s'écrasent jamais entre elles.

Exemples — un par section du notebook 10_embeddings.ipynb :
  S2  corpus global        python src/embedding.py tfidf
                           python src/embedding.py codebert mps
  S3  un langage           python src/embedding.py tfidf cpu "C++"
                           python src/embedding.py codebert mps "C++"
  S4  un problème          python src/embedding.py list D
                           python src/embedding.py tfidf cpu "" p02616
                           python src/embedding.py codebert mps "" p02616
  S5  problème × langage   python src/embedding.py tfidf cpu "Python" p02616
                           python src/embedding.py codebert mps "Python" p02616
"""


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "tfidf"

    if method in ("help", "-h", "--help"):
        print(HELP_TEXT)
        sys.exit(0)

    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import (
        load_atcoder_problems,
        OUT_USER_PROFILES,
    )
    from difficulty_labeler import label_abc_problems

    df_atcoder = load_atcoder_problems()
    df_abc     = label_abc_problems(df_atcoder)
    abc_ids    = df_abc["problem_id"].to_list()
    df_users   = pl.read_csv(OUT_USER_PROFILES)

    if method == "list":
        # Affiche les problem_ids disponibles avec leur difficulté et nb de soumissions dans metadata
        diff_filter = sys.argv[2] if len(sys.argv) > 2 else None
        rows = df_abc.filter(pl.col("difficulty").is_in(TARGET_DIFFICULTIES))
        if diff_filter:
            rows = rows.filter(pl.col("difficulty") == diff_filter.upper())
        counts = []
        for row in rows.iter_rows(named=True):
            fp = METADATA_DIR / f"{row['problem_id']}.csv"
            if fp.exists():
                try:
                    n = pl.scan_csv(str(fp)).select(pl.len()).collect().item()
                    counts.append((n, row["problem_id"], row["difficulty"]))
                except Exception:
                    pass
        counts.sort(reverse=True)
        print(f"\n{'problem_id':12s}  {'diff':4s}  submissions")
        print("-" * 35)
        for n, pid, diff in counts[:30]:
            print(f"{pid:12s}  {diff:4s}  {n:,}")
        sys.exit(0)

    # "mps" for Apple Silicon, "cuda" for NVIDIA, "cpu" as fallback
    device     = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    # e.g. "C++" to restrict to C++ variants only (catches C++14, C++17, C++20…)
    language   = sys.argv[3] if len(sys.argv) > 3 else None
    # e.g. "p02402" to restrict to a single problem
    problem_id = sys.argv[4] if len(sys.argv) > 4 else None

    run_pipeline(abc_ids, df_abc, df_users,
                 method=method, device=device,
                 language=language, problem_id=problem_id)
