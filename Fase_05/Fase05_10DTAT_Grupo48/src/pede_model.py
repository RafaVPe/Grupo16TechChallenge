"""
Modelo supervisionado: probabilidade de **defasagem** (defasagem < 0).

Não utiliza ``ian`` nem ``defasagem`` como features (evita vazamento direto do alvo).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.pede_cleaning import PROJECT_ROOT, build_unified, _fase_series_to_int64

NUMERIC_FEATURES: list[str] = [
    "idade_referencia",
    "fase",
    "ano_ingresso",
    "cg",
    "cf",
    "ct",
    "n_avaliacoes",
    "iaa",
    "ieg",
    "ips",
    "ipp",
    "ida",
    "mat",
    "por",
    "ing",
    "ipv",
    "inde_hist_22",
    "inde_hist_23",
    "ano_cohorte",
]

CATEGORICAL_FEATURES: list[str] = ["genero"]


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Monta X, y com alvo binário defasagem < 0."""
    work = df.copy()
    work["fase"] = _fase_series_to_int64(work["fase"])
    work["__y"] = (work["defasagem"] < 0).astype(int)
    X = work[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype(str).replace({"": "Desconhecido"})
    y = work["__y"].values
    return X, y


def make_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("prep", pre), ("clf", clf)])


@dataclass
class TrainResult:
    pipeline: Pipeline
    metrics: dict[str, Any]
    defaults_numeric: dict[str, float]
    defaults_cat: dict[str, str]


def train_risk_model(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42) -> TrainResult:
    X, y = build_xy(df)
    mask = X[NUMERIC_FEATURES].notna().any(axis=1)
    X, y = X.loc[mask], y[mask.values]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pipe = make_pipeline()
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "classification_report": classification_report(y_test, pred, digits=3),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positivos_rate": float(y.mean()),
    }
    defaults_numeric = X_train[NUMERIC_FEATURES].median(numeric_only=True).to_dict()
    defaults_cat = {c: (X_train[c].mode().iloc[0] if len(X_train[c].mode()) else "Desconhecido") for c in CATEGORICAL_FEATURES}
    return TrainResult(pipeline=pipe, metrics=metrics, defaults_numeric=defaults_numeric, defaults_cat=defaults_cat)


def save_bundle(path: Path, result: TrainResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": result.pipeline,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "defaults_numeric": result.defaults_numeric,
        "defaults_cat": result.defaults_cat,
        "metrics": result.metrics,
        "target_definition": "P(defasagem < 0) — defasagem é o D (fase efetiva − ideal) na base harmonizada.",
    }
    joblib.dump(bundle, path)


def default_parquet_path(root: Path | None = None) -> Path:
    r = root or PROJECT_ROOT
    return r / "data_processed" / "pede_unificado.parquet"


def default_model_path(root: Path | None = None) -> Path:
    r = root or PROJECT_ROOT
    return r / "App" / "models" / "risk_defasagem.joblib"


def ensure_parquet(root: Path | None = None) -> Path:
    """Gera o parquet unificado a partir dos CSVs, se ainda não existir."""
    root = root or PROJECT_ROOT
    pq = default_parquet_path(root)
    if not pq.exists():
        pq.parent.mkdir(parents=True, exist_ok=True)
        build_unified(root=root, save_parquet=pq)
    return pq


def ensure_model_saved(root: Path | None = None, *, force: bool = False) -> Path:
    """
    Garante parquet + arquivo ``risk_defasagem.joblib``.

    Usado pelo Streamlit na primeira execução (avaliador não precisa rodar script manual).
    Com ``force=True``, apaga o joblib e treina de novo.
    """
    root = root or PROJECT_ROOT
    ensure_parquet(root)
    out = default_model_path(root)
    if out.exists() and not force:
        try:
            b = joblib.load(out)
            if b.get("numeric_features") != NUMERIC_FEATURES or b.get("categorical_features") != CATEGORICAL_FEATURES:
                force = True
        except Exception:
            force = True
    if force and out.exists():
        out.unlink()
    if out.exists() and not force:
        return out
    df = pd.read_parquet(default_parquet_path(root))
    res = train_risk_model(df)
    save_bundle(out, res)
    return out


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    path = path or default_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {path}. Abra o app Streamlit (gera automaticamente) ou rode: python scripts/train_model.py"
        )
    return joblib.load(path)


def predict_row(bundle: dict[str, Any], row: dict[str, Any]) -> float:
    """Retorna probabilidade da classe positiva (defasagem < 0)."""
    cols = bundle["numeric_features"] + bundle["categorical_features"]
    X = pd.DataFrame([{k: row.get(k, np.nan) for k in cols}])
    return float(bundle["pipeline"].predict_proba(X)[0, 1])
