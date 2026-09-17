"""
fresh_eval_probe.py — "Real evaluation" demo (RQ2, post-28-août extension).

Motivation : la CV a 5 plis prouve deja la generalisation (chaque pli teste sur
des donnees jamais vues a l'entrainement, pour ce pli). Ce que la CV ne fait
jamais : produire un artefact "sonde entrainee" persiste et l'appliquer a du
code jamais touche par AUCUNE etape du corpus existant (pas juste jamais vu a
l'entrainement d'un pli — jamais tire du tout). C'est un test d'integration
bout-en-bout du pipeline reel, pas une meilleure estimation statistique
(un seul tirage est plus bruite qu'une moyenne a 5 plis) — les deux coexistent,
celui-ci ne remplace pas la CV.

Pour chaque probleme cible :
  1. Entraine + sauvegarde (joblib) une sonde GraphCodeBERT-ast sur la TOTALITE
     du corpus existant (pas 80% — ce qu'un deploiement reel utiliserait).
  2. Tire N soumissions fraiches par verdict, exclues du corpus existant par
     submission_id (pas juste une autre graine aleatoire — exclusion garantie).
  3. Les embed via le pipeline reel (embed_graphcodebert_ast, vrai DFG).
  4. Predit avec la sonde figee, compare au vrai verdict.

Les 4 verdicts (AC/RE/TLE/WA) sont toujours tentes pour chaque probleme — pas
de restriction figee a la main. Si le pool frais d'une classe est quasi/tout
a fait epuise (ex. TLE pour p02659 : 7 exemples au total, deja tous dans le
corpus existant), la boucle le decouvre elle-meme, l'affiche, et continue avec
ce qu'il y a (voir "ATTENTION" dans la sortie) — au lieu qu'on l'exclue a
l'avance dans le code.

Usage (env codenet, memes deps que embedding.py), problem_id et n_per_class
en positionnel comme embedding.py :
    python src/fresh_eval_probe.py                        # p02659,p02658 x 100/classe (defaut)
    python src/fresh_eval_probe.py p02922 50               # un seul probleme, 50/classe
    SEED=7 python src/fresh_eval_probe.py                  # autre tirage de soumissions fraiches
    PROBE_MODEL=mlp python src/fresh_eval_probe.py         # MLPClassifier au lieu de LogisticRegression

PROBE_MODEL choisit le classifieur applique APRES l'embedding -- n'affecte pas
quelles soumissions sont tirees ni leur embedding, donc a SEED egale, un
deuxieme run avec un autre PROBE_MODEL reutilise directement le tirage et les
embeddings deja calcules au lieu de refaire la partie lente (lecture code +
GraphCodeBERT-ast) juste pour comparer les classifieurs.
"""
import os
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import OUT_USER_PROFILES, load_atcoder_problems
from difficulty_labeler import label_abc_problems
from embedding import _load_eligible, embed_graphcodebert_ast, load_codes

EMB_DIR = Path("data/processed/embeddings")
OUT_DIR = Path("data/processed/fresh_eval")
EXISTING_TAG = "graphcodebert_ast_python_multi12"  # corpus deja embedded, base d'entrainement
ALL_CLASSES = ["AC", "RE", "TLE", "WA"]  # toujours tentees -- voir la note dans la boucle
# SEED controle QUELLES soumissions fraiches sont tirees -- change-la pour verifier
# que 77.7%/61.8% ne sont pas juste ce tirage precis (deterministe sinon, voir
# la conversation) :  SEED=7 python src/fresh_eval_probe.py
SEED = int(os.environ.get("SEED", 1337))
# Les resultats sont tagges par seed pour ne jamais ecraser un run precedent.
RUN_TAG = f"_seed{SEED}" if SEED != 1337 else ""

PROBE_MODEL = os.environ.get("PROBE_MODEL", "logreg")
if PROBE_MODEL not in ("logreg", "mlp"):
    raise ValueError(f"PROBE_MODEL='{PROBE_MODEL}' invalide -- 'logreg' ou 'mlp' attendu.")
MODEL_TAG = "" if PROBE_MODEL == "logreg" else f"_{PROBE_MODEL}"


class _StringLabelMLP:
    """
    Enveloppe MLPClassifier + un LabelEncoder pour que .fit/.predict prennent
    et rendent les labels d'origine (AC/WA/TLE/RE) directement, comme
    LogisticRegression le fait deja nativement.

    Necessaire a cause d'un bug/limite reelle de sklearn (verifie par une repro
    minimale, sklearn 1.8.0) : MLPClassifier(early_stopping=True) plante sur des
    labels de type string avec `TypeError: ufunc 'isnan' not supported...` --
    son score de validation interne (utilise pour decider quand arreter) appelle
    np.isnan(y_pred), qui ne marche que sur des tableaux numeriques. Encoder en
    entiers avant fit(), decoder apres predict(), evite le probleme entierement.
    """
    def __init__(self, **mlp_kwargs):
        self.pipeline = make_pipeline(StandardScaler(), MLPClassifier(**mlp_kwargs))
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        self.pipeline.fit(X, self.label_encoder.fit_transform(y))
        return self

    def predict(self, X):
        return self.label_encoder.inverse_transform(self.pipeline.predict(X))


def _build_classifier():
    if PROBE_MODEL == "mlp":
        # Petit reseau, regularise vu la taille des donnees (~1500-2000/probleme) :
        # une seule couche cachee, alpha (L2) monte par rapport au defaut, et
        # early_stopping reserve 15% des donnees en interne pour arreter des que
        # ca stagne -- le vrai garde-fou contre le surapprentissage ici.
        return _StringLabelMLP(
            hidden_layer_sizes=(64,), alpha=1e-2, early_stopping=True,
            validation_fraction=0.15, max_iter=1000, random_state=42,
        )
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

# problem_id (liste separee par des virgules) et n_per_class en CLI, comme
# embedding.py -- pas fige en dur, pas de restriction de classes en dur non
# plus : on tente toujours les 4 verdicts, la boucle plus bas gere elle-meme
# le cas d'une classe quasi vide (voir p02659/TLE) au lieu qu'on l'exclue a la main.
PROBLEM_IDS = (sys.argv[1] if len(sys.argv) > 1 else "p02659,p02658").split(",")
N_PER_CLASS = int(sys.argv[2]) if len(sys.argv) > 2 else 100


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_atcoder = load_atcoder_problems()
    df_abc = label_abc_problems(df_atcoder)
    abc_ids = df_abc["problem_id"].to_list()
    df_users = pl.read_csv(OUT_USER_PROFILES)

    emb_existing = np.load(EMB_DIR / f"embeddings_{EXISTING_TAG}.npy")
    meta_existing = pl.read_csv(EMB_DIR / f"metadata_{EXISTING_TAG}.csv")

    summary = []
    for pid in PROBLEM_IDS:
        print(f"\n{'=' * 60}\n{pid}\n{'=' * 60}")

        # 1. Entrainer + sauvegarder la sonde sur TOUT le corpus existant
        idx_train = np.where((meta_existing["problem_id"] == pid).to_numpy())[0]
        X_train = emb_existing[idx_train]
        y_train = meta_existing["status_code"].to_numpy()[idx_train]
        print(f"Entrainement ({PROBE_MODEL}) sur {len(idx_train)} exemples existants "
              f"({dict(zip(*np.unique(y_train, return_counts=True)))})")

        clf = _build_classifier()
        clf.fit(X_train, y_train)
        clf_path = OUT_DIR / f"probe_graphcodebert_ast_{pid}{MODEL_TAG}.joblib"
        joblib.dump(clf, clf_path)
        print(f"Sonde sauvegardee : {clf_path}")

        # 2-3. Pool frais + embedding -- reutilises tels quels si deja calcules
        # pour ce (pid, seed) : le tirage et l'embedding ne dependent pas de
        # PROBE_MODEL, inutile de refaire la partie lente juste pour comparer
        # les classifieurs.
        draw_path = OUT_DIR / f"fresh_draw_{pid}{RUN_TAG}.csv"
        emb_path = OUT_DIR / f"fresh_embeddings_{pid}{RUN_TAG}.npy"
        if draw_path.exists() and emb_path.exists():
            df_fresh_valid = pl.read_csv(draw_path)
            emb_fresh = np.load(emb_path)
            print(f"Tirage + embeddings deja sur disque ({draw_path.name}) -- reutilises tels quels.")
        else:
            # Pool frais = eligible - deja utilise. On tente les 4 verdicts --
            # pas de restriction a la main : si une classe est quasi/totalement
            # vide dans le pool frais (ex. TLE pour p02659), on le decouvre ici,
            # on l'affiche, et on continue avec ce qu'il y a (potentiellement 0).
            df_elig = _load_eligible(abc_ids, df_abc, df_users, language="Python", problem_id=pid)
            already_ids = set(meta_existing.filter(pl.col("problem_id") == pid)["submission_id"].to_list())
            df_fresh_pool = df_elig.filter(~pl.col("submission_id").is_in(list(already_ids)))

            rng = random.Random(SEED)
            parts = []
            for status in ALL_CLASSES:
                pool = df_fresh_pool.filter(pl.col("status_code") == status)
                n = min(N_PER_CLASS, pool.height)
                if n < N_PER_CLASS:
                    print(f"  ATTENTION : seulement {n} dispo pour {status} (voulu {N_PER_CLASS})")
                if n == 0:
                    continue
                picked = np.array(rng.sample(range(pool.height), n), dtype=int)
                parts.append(pool[picked])
            df_fresh = pl.concat(parts)
            print(f"{df_fresh.height} soumissions fraiches tirees (jamais dans le corpus existant).")

            codes, valid_mask = load_codes(df_fresh)
            df_fresh_valid = df_fresh.filter(pl.Series("valid", valid_mask))
            print(f"Embedding {len(codes)} soumissions (graphcodebert_ast, device=mps)...")
            emb_fresh = embed_graphcodebert_ast(codes, dfg_lang="python", device="mps")

            df_fresh_valid.write_csv(draw_path)
            np.save(emb_path, emb_fresh)

        # 4. Predire avec la sonde figee, comparer au vrai verdict
        y_true = df_fresh_valid["status_code"].to_numpy()
        y_pred = clf.predict(emb_fresh)
        acc = float((y_pred == y_true).mean())
        print(f"\nAccuracy sur donnees fraiches (jamais vues, jamais embedded avant) : {acc:.1%}")

        out = df_fresh_valid.with_columns([
            pl.Series("predicted", y_pred),
            pl.Series("correct", y_pred == y_true),
        ])
        eval_path = OUT_DIR / f"fresh_eval_{pid}{MODEL_TAG}{RUN_TAG}.csv"
        out.write_csv(eval_path)
        print(f"Sauvegarde : {eval_path}")

        summary.append({"problem_id": pid, "model": PROBE_MODEL, "seed": SEED,
                         "n_fresh": df_fresh_valid.height, "fresh_accuracy": round(acc, 4)})

    print(f"\n{'=' * 60}\nResume (model={PROBE_MODEL}, seed={SEED})\n{'=' * 60}")
    print(pl.DataFrame(summary))


if __name__ == "__main__":
    main()
