"""
Bootstrap compartilhado do Streamlit (parquet + modelo).

Usar o **mesmo** ``@st.cache_resource`` em todas as páginas para treinar só uma vez por worker.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


@st.cache_resource(show_spinner="Preparando base harmonizada e modelo…")
def ensure_streamlit_artifacts(project_root: Path) -> None:
    """Gera ``pede_unificado.parquet`` e ``risk_defasagem.joblib`` se ainda não existirem."""
    from src.pede_model import ensure_model_saved

    ensure_model_saved(project_root, force=False)
