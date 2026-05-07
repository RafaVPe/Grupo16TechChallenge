"""Treina e salva o modelo (opcional: o Streamlit já gera na primeira abertura). Execute na raiz do projeto."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pede_model import default_model_path, ensure_model_saved, load_bundle


def main() -> None:
    ap = argparse.ArgumentParser(description="Treina risk_defasagem.joblib")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Apaga o joblib existente e treina de novo.",
    )
    args = ap.parse_args()
    out = ensure_model_saved(ROOT, force=args.force)
    print("Modelo disponível em:", out)
    b = load_bundle(default_model_path(ROOT))
    print("ROC-AUC (holdout):", b["metrics"].get("roc_auc"))


if __name__ == "__main__":
    main()
