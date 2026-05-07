"""
Carrega e harmoniza os CSVs PEDE 2022, 2023 e 2024.

Regras principais
-----------------
- Chave lógica de painel: (``ra``, ``ano_cohorte``). Não há duplicata de RA
  dentro do mesmo arquivo; cada ano vira uma linha distinta.
- Colunas duplicadas no CSV (ex.: dois ``Destaque IPV``; dois ``Ativo/ Inativo``)
  são fundidas por coalesce (primeiro valor não nulo).
- ``inde_cohorte`` é o INDE do ano de referência da linha (ex.: INDE 2024 na
  base 2024). Colunas ``INDE 22`` / ``INDE 23`` preservadas como histórico
  com nomes estáveis (podem repetir valores já vistos em anos anteriores —
  isso é esperado em painel, não é erro de merge).
- Números com vírgula decimal (pt-BR) convertidos para float.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FILES: dict[int, str] = {
    2022: "BASE DE DADOS PEDE 2024 - DATATHON - PEDE2022.csv",
    2023: "BASE DE DADOS PEDE 2024 - DATATHON - PEDE2023.csv",
    2024: "BASE DE DADOS PEDE 2024 - DATATHON - PEDE2024.csv",
}


def _strip_bom(s: str) -> str:
    return str(s).replace("\ufeff", "").strip()


def brazilian_to_float(value: Any) -> float | np.floating:
    """Converte string numérica pt-BR (vírgula decimal) em float; vazio vira NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
        return float(value)
    s = _strip_bom(value)
    if s == "" or s.lower() in {"nan", "none", "na", "n/a"}:
        return np.nan
    s = s.replace('"', "").replace("'", "")
    s = s.replace(".", "").replace(",", ".") if re.search(r",\d+$", s) else s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Une colunas com mesmo nome base (ex.: pandas renomeou para ``.1``)."""
    out = df.copy()
    # Destaque IPV duplicado em 2023
    cols = list(out.columns)
    if "Destaque IPV.1" in cols and "Destaque IPV" in cols:
        out["Destaque IPV"] = out["Destaque IPV"].where(
            out["Destaque IPV"].notna() & (out["Destaque IPV"].astype(str).str.strip() != ""),
            out["Destaque IPV.1"],
        )
        out = out.drop(columns=["Destaque IPV.1"])
    # Ativo/Inativo duplicado em 2024
    if "Ativo/ Inativo.1" in out.columns and "Ativo/ Inativo" in out.columns:
        out["Ativo/ Inativo"] = out["Ativo/ Inativo"].where(
            out["Ativo/ Inativo"].notna()
            & (out["Ativo/ Inativo"].astype(str).str.strip() != ""),
            out["Ativo/ Inativo.1"],
        )
        out = out.drop(columns=["Ativo/ Inativo.1"])
    return out


def _norm_col(name: str) -> str:
    return (
        _strip_bom(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("º", "o")
        .replace("°", "o")
    )


def load_raw_csv(path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False)
    df = _coalesce_duplicate_columns(df)
    # normaliza nomes internos para merge
    df.columns = [_norm_col(c) for c in df.columns]
    return df


def _parse_genero(s: Any) -> str:
    t = _strip_bom(s).lower()
    if t in {"menina", "feminino", "f"}:
        return "Feminino"
    if t in {"menino", "masculino", "m"}:
        return "Masculino"
    if t == "":
        return ""
    return _strip_bom(s)


def _harmonize_2022(df: pd.DataFrame) -> pd.DataFrame:
    h = pd.DataFrame()
    h["ra"] = df["ra"].astype(str).str.strip()
    h["ano_cohorte"] = 2022
    h["fase"] = df["fase"].astype(str).str.strip()
    h["turma"] = df.get("turma", "")
    h["nome"] = df.get("nome", "")
    h["ano_nascimento"] = df.get("ano_nasc", "").map(brazilian_to_float)
    h["data_nasc"] = pd.NaT
    h["idade_referencia"] = df.get("idade_22", "").map(brazilian_to_float)
    h["genero"] = df["gênero"].map(_parse_genero)
    h["ano_ingresso"] = df.get("ano_ingresso", "").map(brazilian_to_float)
    h["instituicao_ensino"] = df["instituição_de_ensino"]
    h["escola"] = ""
    h["status_aluno"] = ""
    for p in ["pedra_20", "pedra_21", "pedra_22"]:
        h[p] = df.get(p, "")
    h["pedra_23"] = ""
    h["pedra_atual"] = df.get("pedra_22", "")
    h["inde_cohorte"] = df.get("inde_22", "").map(brazilian_to_float)
    h["inde_hist_22"] = df.get("inde_22", "").map(brazilian_to_float)
    h["inde_hist_23"] = np.nan
    h["inde_hist_24"] = np.nan
    h["cg"] = df.get("cg", "").map(brazilian_to_float)
    h["cf"] = df.get("cf", "").map(brazilian_to_float)
    h["ct"] = df.get("ct", "").map(brazilian_to_float)
    h["n_avaliacoes"] = df.get("no_av", "").map(brazilian_to_float)
    for ind in ["iaa", "ieg", "ips", "ida", "ipv", "ian"]:
        h[ind] = df.get(ind, "").map(brazilian_to_float)
    h["ipp"] = np.nan  # não consta no layout 2022 do CSV fornecido
    h["mat"] = df.get("matem", "").map(brazilian_to_float)
    h["por"] = df.get("portug", "").map(brazilian_to_float)
    h["ing"] = df["inglês"].map(brazilian_to_float)
    h["indicado"] = df.get("indicado", "")
    h["atingiu_pv"] = df.get("atingiu_pv", "")
    h["fase_ideal"] = df.get("fase_ideal", "")
    h["defasagem"] = df.get("defas", "").map(brazilian_to_float)
    h["rec_psicologia"] = df.get("rec_psicologia", "")
    h["destaque_ieg"] = df.get("destaque_ieg", "")
    h["destaque_ida"] = df.get("destaque_ida", "")
    h["destaque_ipv"] = df.get("destaque_ipv", "")
    return h


def _harmonize_2023_2024(df: pd.DataFrame, ano: int) -> pd.DataFrame:
    h = pd.DataFrame()
    h["ra"] = df["ra"].astype(str).str.strip()
    h["ano_cohorte"] = ano
    h["fase"] = df["fase"].astype(str).str.strip()
    h["turma"] = df.get("turma", "")
    h["nome"] = df.get("nome_anonimizado", "")
    h["data_nasc"] = pd.to_datetime(df.get("data_de_nasc", ""), errors="coerce", format="mixed")
    h["ano_nascimento"] = h["data_nasc"].dt.year
    h["idade_referencia"] = df.get("idade", "").map(brazilian_to_float)
    h["genero"] = df["gênero"].map(_parse_genero)
    h["ano_ingresso"] = df.get("ano_ingresso", "").map(brazilian_to_float)
    h["instituicao_ensino"] = df["instituição_de_ensino"]
    h["escola"] = df.get("escola", "") if ano == 2024 else ""
    h["status_aluno"] = df.get("ativo__inativo", "") if ano == 2024 else ""

    for p in ["pedra_20", "pedra_21", "pedra_22", "pedra_23"]:
        h[p] = df.get(p, "")
    h["pedra_atual"] = df.get(f"pedra_{ano}", "")

    col_inde = f"inde_{ano}"
    h["inde_cohorte"] = df.get(col_inde, "").map(brazilian_to_float)
    h["inde_hist_22"] = df.get("inde_22", "").map(brazilian_to_float)
    h["inde_hist_23"] = df.get("inde_23", "").map(brazilian_to_float)
    h["inde_hist_24"] = df.get("inde_24", "").map(brazilian_to_float) if "inde_24" in df.columns else np.nan

    h["cg"] = df.get("cg", "").map(brazilian_to_float)
    h["cf"] = df.get("cf", "").map(brazilian_to_float)
    h["ct"] = df.get("ct", "").map(brazilian_to_float)
    h["n_avaliacoes"] = df.get("no_av", "").map(brazilian_to_float)
    for ind in ["iaa", "ieg", "ips", "ipp", "ida", "ipv", "ian"]:
        h[ind] = df.get(ind, "").map(brazilian_to_float)
    h["mat"] = df.get("mat", "").map(brazilian_to_float)
    h["por"] = df.get("por", "").map(brazilian_to_float)
    h["ing"] = df.get("ing", "").map(brazilian_to_float)
    h["indicado"] = df.get("indicado", "")
    h["atingiu_pv"] = df.get("atingiu_pv", "")
    h["fase_ideal"] = df.get("fase_ideal", "")
    h["defasagem"] = df.get("defasagem", "").map(brazilian_to_float)
    h["rec_psicologia"] = df.get("rec_psicologia", "")
    h["destaque_ieg"] = df.get("destaque_ieg", "")
    h["destaque_ida"] = df.get("destaque_ida", "")
    h["destaque_ipv"] = df.get("destaque_ipv", "")
    return h


def harmonize_year(df_raw: pd.DataFrame, ano: int) -> pd.DataFrame:
    if ano == 2022:
        return _harmonize_2022(df_raw)
    if ano in (2023, 2024):
        return _harmonize_2023_2024(df_raw, ano)
    raise ValueError(f"Ano não suportado: {ano}")


def build_unified(
    root: Path | None = None,
    files: dict[int, str] | None = None,
    save_parquet: Path | None = None,
) -> pd.DataFrame:
    """
    Lê os três CSVs, harmoniza e empilha.

    Parameters
    ----------
    save_parquet
        Se informado, grava o resultado (ex.: ``data_processed/pede_unificado.parquet``).
    """
    root = root or PROJECT_ROOT
    files = files or DEFAULT_FILES
    parts: list[pd.DataFrame] = []
    for ano, fname in sorted(files.items()):
        path = root / fname
        if not path.exists():
            raise FileNotFoundError(path)
        raw = load_raw_csv(path)
        parts.append(harmonize_year(raw, ano))
    out = pd.concat(parts, ignore_index=True)

    # validação de unicidade (RA, ano)
    dup_mask = out.duplicated(subset=["ra", "ano_cohorte"], keep=False)
    if dup_mask.any():
        bad = out.loc[dup_mask, ["ra", "ano_cohorte"]].drop_duplicates()
        raise ValueError(f"Chaves duplicadas (ra, ano_cohorte):\n{bad.head(20)}")

    if save_parquet is not None:
        save_parquet.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(save_parquet, index=False)

    return out


def cleaning_report(df: pd.DataFrame) -> dict[str, Any]:
    """Resumo para UI / notebook."""
    by_year = df.groupby("ano_cohorte").size().to_dict()
    n_ra = df["ra"].nunique()
    n_rows = len(df)
    return {
        "n_rows": n_rows,
        "n_ra_distintos": n_ra,
        "linhas_por_ano": by_year,
        "cols": list(df.columns),
        "na_rate_inde_cohorte": float(df["inde_cohorte"].isna().mean()),
        "na_rate_ian": float(df["ian"].isna().mean()),
    }
