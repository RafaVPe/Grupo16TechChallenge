"""
Página inicial — DATATHON (Fase 05) · Case PEDE Passos Mágicos.

Execute na raiz do projeto:
  python -m streamlit run App/DATATHON_Inicio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pede_cleaning import PROJECT_ROOT
from src.streamlit_bootstrap import ensure_streamlit_artifacts

st.set_page_config(
    page_title="DATATHON — Fase 05 | Início",
    layout="wide",
)

ensure_streamlit_artifacts(PROJECT_ROOT)

_HERO = """
<div style="
  background: linear-gradient(125deg, #0f172a 0%, #1e3a8a 38%, #0e7490 72%, #134e4a 100%);
  padding: 1.75rem 1.5rem 1.5rem;
  border-radius: 14px;
  margin-bottom: 1.25rem;
  box-shadow: 0 8px 32px rgba(15, 23, 42, 0.45);
  border: 1px solid rgba(148, 163, 184, 0.25);
">
  <p style="
    margin: 0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-weight: 800;
    font-size: clamp(1.85rem, 4vw, 2.65rem);
    letter-spacing: 0.18em;
    color: #f8fafc;
    text-align: center;
    text-shadow: 0 2px 18px rgba(0,0,0,0.35);
  ">DATATHON</p>
  <p style="
    margin: 0.5rem 0 0;
    text-align: center;
    color: #bae6fd;
    font-size: 1.05rem;
    font-weight: 500;
  ">Fase 05 · Pós-graduação em <strong style="color:#ecfeff;">Data Analytics</strong></p>
  <p style="
    margin: 0.35rem 0 0;
    text-align: center;
    color: #94a3b8;
    font-size: 0.9rem;
  ">Case <strong style="color:#e2e8f0;">Passos Mágicos — PEDE</strong> · Base harmonizada 2022–2024</p>
</div>
"""

st.markdown(_HERO, unsafe_allow_html=True)

st.markdown("### Sobre o trabalho")
st.markdown(
    "Este projeto entrega a **análise exploratória e prescritiva** do case PEDE (Passos Mágicos), com dados "
    "unificados por **(RA, ano_cohorte)**, **dez perguntas de negócio** em painel interativo, **Insights** com "
    "filtros e leituras automáticas, e um **modelo supervisionado** (Random Forest) que estima a probabilidade de "
    "**defasagem negativa** (fase efetiva abaixo da ideal), com simulador de cenários."
)

st.markdown("### Navegação")
st.caption("Use os botões abaixo ou o menu **Pages** na barra lateral do Streamlit.")

# Três botões no mesmo estilo (contorno); vermelho só no hover — escopo só na área principal (não afeta a sidebar).
# Seletores duplos: `kind` (algumas versões) e `data-testid` (BaseButton mais recente).
st.markdown(
    """
<style>
section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] button[kind="secondary"],
section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] button[data-testid="baseButton-secondary"] {
  background-color: transparent !important;
  color: #fafafa !important;
  border: 1px solid rgba(250, 250, 250, 0.42) !important;
  transition: background-color 0.16s ease, border-color 0.16s ease, color 0.16s ease !important;
}
section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover,
section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] button[data-testid="baseButton-secondary"]:hover {
  background-color: #ff4b4b !important;
  border-color: #ff4b4b !important;
  color: #ffffff !important;
}
section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] button[kind="secondary"]:focus-visible,
section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] button[data-testid="baseButton-secondary"]:focus-visible {
  box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.55) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

b1, b2, b3 = st.columns(3)
with b1:
    if st.button("PEDE — 10 perguntas", type="secondary", use_container_width=True, help="Gráficos e textos das perguntas 1 a 10"):
        st.switch_page("pages/1_PEDE_10_perguntas.py")
with b2:
    if st.button("Insights", type="secondary", use_container_width=True, help="Filtros, leituras automáticas e painéis dinâmicos"):
        st.switch_page("pages/2_Insights.py")
with b3:
    if st.button("Modelo / Predição", type="secondary", use_container_width=True, help="Simulador de risco com o modelo treinado"):
        st.switch_page("pages/3_Modelo_predicao.py")

st.divider()
st.markdown("### Equipe")
st.markdown(
    """
- **ANDERSON CORREIA BALBINO** — RM 365550  
- **ELECIDES TEIXEIRA JUNIOR** — RM 366315  
- **GABRIEL MACIEL DOS SANTOS** — RM 365687  
- **RAFAEL VICTOR PEREIRA** — RM 366441  
- **RHÔMULO MOURÃO CAITANO DOS SANTOS** — RM 365205  
"""
)

st.caption("Disciplina / entrega: **Fase 05 — DATATHON** · Pós-graduação em Data Analytics.")
