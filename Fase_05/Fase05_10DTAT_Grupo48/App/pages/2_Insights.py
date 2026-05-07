"""Insights complementares — pergunta 11 + painéis dinâmicos (UI em abas)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.insights_dynamic import aplicar_filtros, gerar_insights
from src.pede_cleaning import PROJECT_ROOT
from src.streamlit_bootstrap import ensure_streamlit_artifacts

ensure_streamlit_artifacts(PROJECT_ROOT)

PROCESSED = PROJECT_ROOT / "data_processed" / "pede_unificado.parquet"

head_l, head_r = st.columns([5, 1])
with head_l:
    st.title("Insights")
with head_r:
    if st.button("Modelo / predição", type="secondary", use_container_width=True, help="Abre a página de inferência com o Random Forest"):
        st.switch_page("pages/3_Modelo_predicao.py")

st.caption(
    "Monte o **recorte** abaixo; **leituras automáticas** e gráficos atualizam na hora. "
    "As abas organizam os painéis por tipo de pergunta de negócio."
)

if not PROCESSED.exists():
    st.error("Não foi possível gerar o parquet. Verifique se os CSVs estão na raiz do projeto.")
    st.stop()

df_full = pd.read_parquet(PROCESSED)

anos_opts = sorted(df_full["ano_cohorte"].dropna().unique().astype(int).tolist())
gen_opts = sorted(df_full["genero"].dropna().astype(str).unique().tolist())
fase_counts = df_full["fase"].astype(str).value_counts()
top_fases = fase_counts.head(35).index.tolist()

# Painel de filtros no corpo da página (sem sidebar) — hierarquia: recorte → aviso fases → leituras → abas
_filter_shell = (
    st.container(border=True) if "border" in inspect.signature(st.container).parameters else st.container()
)
with _filter_shell:
    st.markdown("##### Recorte da análise")
    st.caption("Ano e gênero definem o universo; **fases** refinam onde a defasagem concentra (aba *Risco & fases*).")
    fa, fg, ff = st.columns([1.05, 0.95, 1.25])
    with fa:
        anos = st.multiselect(
            "Ano do painel",
            options=anos_opts,
            default=anos_opts,
            help="Ano da linha no PEDE (`ano_cohorte`: 2022, 2023, 2024). Vazio volta a considerar todos os anos.",
        )
    with fg:
        generos = st.multiselect(
            "Gênero",
            options=gen_opts,
            default=gen_opts,
            help="Vazio volta a considerar todos os gêneros da base.",
        )
    with ff:
        fases_sel = st.multiselect(
            "Fases escolares",
            options=top_fases,
            default=[],
            help=(
                "Sem seleção: **todas** as fases entram no recorte (números mais agregados). "
                "Lista limitada às 35 fases mais frequentes na base unificada."
            ),
        )

fases = fases_sel if fases_sel else None
anos_ef = anos if anos else anos_opts
gen_ef = generos if generos else gen_opts
df = aplicar_filtros(df_full, anos=anos_ef, generos=gen_ef, fases=fases)

if not fases_sel:
    st.warning(
        "**Nenhuma fase selecionada.** O recorte atual inclui **todas** as fases — útil para visão geral, "
        "mas dilui o sinal por ciclo. Para leituras e gráficos mais **acionáveis** (onde a defasagem pesa por etapa), "
        "escolha ao menos **uma fase** em *Fases escolares* acima."
    )

st.subheader("Leituras automáticas")
for texto, tipo in gerar_insights(df, df_full):
    if tipo == "success":
        st.success(texto)
    elif tipo == "warning":
        st.warning(texto)
    elif tipo == "error":
        st.error(texto)
    else:
        st.info(texto)

st.divider()

tab_vis, tab_risco, tab_cruz, tab_extra = st.tabs(
    ["Visão geral", "Risco & fases", "Indicadores cruzados", "Composição"]
)

with tab_vis:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros (filtro)", f"{len(df):,}")
    if len(df):
        c2.metric("Taxa defasagem negativa", f"{(df['defasagem'] < 0).mean():.1%}")
        c3.metric(
            "INDE mediano",
            f"{df['inde_cohorte'].median():.2f}" if df["inde_cohorte"].notna().any() else "—",
        )
        c4.metric("IEG mediano", f"{df['ieg'].median():.2f}" if df["ieg"].notna().any() else "—")
    fig_inde = px.box(
        df.dropna(subset=["inde_cohorte"]),
        x="ano_cohorte",
        y="inde_cohorte",
        color="genero",
        title="Distribuição do INDE por ano do painel e gênero",
        labels={"inde_cohorte": "INDE", "ano_cohorte": "Ano do painel"},
    )
    fig_inde.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig_inde, width="stretch")
    fig_ie = px.histogram(
        df.dropna(subset=["ieg"]),
        x="ieg",
        color="ano_cohorte",
        nbins=30,
        barmode="overlay",
        opacity=0.65,
        title="Distribuição de IEG (sobreposição por ano)",
    )
    fig_ie.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig_ie, width="stretch")

with tab_risco:
    d = df.copy()
    d["defas_neg"] = (d["defasagem"] < 0).astype(int)
    agg = (
        d.groupby(["ano_cohorte", "fase"], as_index=False)["defas_neg"]
        .mean()
        .sort_values(["ano_cohorte", "defas_neg"], ascending=[True, False])
    )
    fig1 = px.bar(
        agg.head(36),
        x="fase",
        y="defas_neg",
        color="ano_cohorte",
        barmode="group",
        title="Taxa de defasagem negativa por fase (até 36 linhas fase×ano)",
        labels={"defas_neg": "Proporção", "fase": "Fase"},
    )
    fig1.update_layout(template="plotly_dark", height=520, xaxis_tickangle=-45)
    st.plotly_chart(fig1, width="stretch")

    d2 = df.dropna(subset=["ian"]).copy()
    d2["faixa_ian"] = pd.cut(
        d2["ian"],
        bins=[-0.1, 4.0, 9.0, 11.0],
        labels=["Severa/mod. (<4)", "Moderada (4–9)", "Em fase (≥9)"],
    )
    fig_stack = px.histogram(
        d2.dropna(subset=["faixa_ian"]),
        x="ano_cohorte",
        color="faixa_ian",
        barmode="stack",
        title="Volume por ano e faixa aproximada de IAN",
    )
    fig_stack.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_stack, width="stretch")

with tab_cruz:
    num_cols = [
        c
        for c in ["iaa", "ieg", "ips", "ipp", "ida", "ian", "ipv", "inde_cohorte", "mat", "por", "ing"]
        if c in df.columns
    ]
    sub = df[num_cols].dropna(axis=0, how="all")
    if len(sub) > 10 and len(num_cols) > 2:
        corr = sub.corr(numeric_only=True).abs()
        fig2 = px.imshow(
            corr,
            text_auto=".2f",
            title="|Correlação de Pearson| no recorte filtrado",
            color_continuous_scale="Blues",
            aspect="auto",
        )
        fig2.update_layout(template="plotly_dark", height=560)
        st.plotly_chart(fig2, width="stretch")
    else:
        st.warning("Poucos dados numéricos no filtro. Afrouxe os filtros para ver a matriz.")

    sc = df.dropna(subset=["iaa", "ida"]).copy()
    sc["defas_flag"] = (sc["defasagem"] < 0).map({True: "Defasagem negativa", False: "Em fase / adiantado"})
    if len(sc) > 15:
        sample = sc.sample(min(2500, len(sc)), random_state=7)
        use_facet = sc["ano_cohorte"].nunique() <= 4
        fig_sc = px.scatter(
            sample,
            x="ida",
            y="iaa",
            color="defas_flag",
            facet_col="ano_cohorte" if use_facet else None,
            title="IAA × IDA (cor = defasagem negativa ou não)",
            labels={"ida": "IDA", "iaa": "IAA"},
            height=480 if use_facet else 420,
        )
        fig_sc.update_layout(template="plotly_dark")
        st.plotly_chart(fig_sc, width="stretch")
    else:
        st.info("Poucos pontos para o dispersograma IAA×IDA.")

with tab_extra:
    ped = (
        df.groupby(["ano_cohorte", "pedra_atual"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values("n", ascending=False)
    )
    if not ped.empty and ped["pedra_atual"].notna().any():
        fig_t = px.treemap(
            ped,
            path=["ano_cohorte", "pedra_atual"],
            values="n",
            title="Composição: registros por ano e pedra atual",
        )
        fig_t.update_layout(template="plotly_dark", height=520)
        st.plotly_chart(fig_t, width="stretch")
    trend = df.groupby("ano_cohorte", as_index=False)[["ida", "ieg", "inde_cohorte"]].mean()
    if not trend.empty and trend["ano_cohorte"].notna().all():
        long = trend.melt(id_vars="ano_cohorte", var_name="indicador", value_name="media")
        fig_m = px.line(
            long,
            x="ano_cohorte",
            y="media",
            color="indicador",
            markers=True,
            title="Médias no filtro: IDA, IEG e INDE por ano",
        )
        fig_m.update_layout(template="plotly_dark", height=400, legend_title_text="Indicador")
        st.plotly_chart(fig_m, width="stretch")

st.divider()
st.markdown(
    "**Como usar:** *Visão geral* para KPIs e distribuições; *Risco & fases* para onde a defasagem concentra; "
    "*Indicadores cruzados* para discussão técnica; *Composição* para mix de pedras. Combine com a **aba 9 (ML)** "
    "e a página **Modelo** para triagem de risco."
)
