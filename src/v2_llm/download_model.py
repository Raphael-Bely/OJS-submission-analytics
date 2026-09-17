"""
download_model.py — Downloads a Hugging Face Hub model snapshot to a local directory.

Requires: pip install huggingface_hub

Usage:
    python src/v2_llm/download_model.py Qwen/Qwen2.5-Coder-7B-Instruct
    python src/v2_llm/download_model.py Qwen/Qwen2.5-Coder-7B-Instruct data/models/my-copy
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODELS_DIR = Path("./data/models")


def download_model(model_name: str, local_dir: str | Path) -> None:
    local_dir = Path(local_dir)
    if local_dir.exists():
        print(f"Le repertoire {local_dir} existe deja. Le telechargement va reprendre ou mettre a jour le contenu existant.")
    else:
        print(f"Telechargement du modele '{model_name}' vers '{local_dir}'...")
    try:
        snapshot_download(repo_id=model_name, local_dir=local_dir)
        print("Telechargement termine avec succes.")
    except Exception as e:
        print(f"Erreur lors du telechargement : {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telecharge un modele depuis Hugging Face.")
    parser.add_argument("model_name", type=str,
                         help="Nom du modele sur le Hub (ex: Qwen/Qwen2.5-Coder-7B-Instruct).")
    parser.add_argument("output_dir", type=str, nargs="?", default=None,
                         help="Repertoire de destination (defaut : data/models/<org>--<modele>).")
    args = parser.parse_args()
    output_dir = args.output_dir or DEFAULT_MODELS_DIR / args.model_name.replace("/", "--")
    download_model(args.model_name, output_dir)
