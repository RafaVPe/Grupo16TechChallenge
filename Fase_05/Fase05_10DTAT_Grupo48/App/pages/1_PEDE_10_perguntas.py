"""
Passos Mágicos — PEDE: dez perguntas (abas) + qualidade da base.

Ponto de entrada do app: `App/DATATHON_Inicio.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pede_analysis import TAB_FUNCS, TAB_TITLES
from src.pede_questions import QUESTIONS_TEXT
from src.pede_cleaning import PROJECT_ROOT, build_unified, cleaning_report, apply_fase_column_normalization
from src.streamlit_bootstrap import ensure_streamlit_artifacts

PROCESSED = PROJECT_ROOT / "data_processed" / "pede_unificado.parquet"

_DATATHON_STRIP = """
<div style="
  background: linear-gradient(125deg, #0f172a 0%, #1e3a8a 40%, #0f766e 100%);
  padding: 1rem 1.25rem;
  border-radius: 12px;
  margin-bottom: 0.75rem;
  border: 1px solid rgba(148, 163, 184, 0.2);
">
  <p style="margin:0; text-align:center; font-weight:800; letter-spacing:0.2em; color:#f1f5f9; font-size:1.35rem;">DATATHON</p>
  <p style="margin:0.25rem 0 0; text-align:center; color:#99f6e4; font-size:0.82rem;">Fase 05 · Data Analytics</p>
</div>
"""


@st.cache_data(show_spinner=True)
def carregar_base() -> pd.DataFrame:
    if PROCESSED.exists():
        return apply_fase_column_normalization(pd.read_parquet(PROCESSED))
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    return build_unified(save_parquet=PROCESSED)


st.set_page_config(
    page_title="PEDE — Passos Mágicos",
    layout="wide",
)

ensure_streamlit_artifacts(PROJECT_ROOT)

st.markdown(_DATATHON_STRIP, unsafe_allow_html=True)
st.caption("Toque no botão abaixo para voltar à **página inicial** (DATATHON · Fase 05).")
if st.button(
    "Abrir página inicial — DATATHON (Fase 05)",
    key="goto_inicio_datathon",
    use_container_width=True,
    help="Redireciona para a tela de apresentação do trabalho e da equipe",
):
    st.switch_page("DATATHON_Inicio.py")

st.title("PEDE — análise (perguntas 1 a 10)")
st.caption("Base harmonizada 2022–2024 · Chave (RA, ano_cohorte) · **Insights** e **Modelo** no menu lateral (Pages).")

with st.sidebar:
    st.header("Dados")
    if st.button("Reprocessar CSV → parquet"):
        if PROCESSED.exists():
            PROCESSED.unlink()
        carregar_base.clear()
        st.success("Cache limpo. Recarregue a página.")
    st.caption("Página inicial: **DATATHON_Inicio** no menu do app ou botão **Início DATATHON** acima.")

df = carregar_base()

with st.expander("Qualidade da base (resumo)", expanded=False):
    rep = cleaning_report(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Linhas", f"{rep['n_rows']:,}")
    c2.metric("RA distintos", f"{rep['n_ra_distintos']:,}")
    c3.metric("NA em INDE (painel)", f"{rep['na_rate_inde_cohorte']:.1%}")
    ra_por_ano = df.groupby("ano_cohorte")["ra"].apply(set).to_dict()
    y22, y23, y24 = ra_por_ano.get(2022, set()), ra_por_ano.get(2023, set()), ra_por_ano.get(2024, set())
    st.table(
        pd.DataFrame(
            [
                {"corte": "2022 ∩ 2023", "RA": len(y22 & y23)},
                {"corte": "2022 ∩ 2024", "RA": len(y22 & y24)},
                {"corte": "2023 ∩ 2024", "RA": len(y23 & y24)},
                {"corte": "Nos 3 anos", "RA": len(y22 & y23 & y24)},
            ]
        )
    )

tabs = st.tabs(TAB_TITLES)
for i, tab in enumerate(tabs):
    with tab:
        st.markdown(QUESTIONS_TEXT[i])
        fig, md = TAB_FUNCS[i](df)
        st.markdown(md)
        st.plotly_chart(fig, width="stretch")
        if i == 8:
            st.divider()
            if st.button(
                "Abrir Modelo / Predição (simulador)",
                type="primary",
                key="q9_switch_modelo",
                use_container_width=True,
            ):
                st.switch_page("pages/3_Modelo_predicao.py")
