"""Probabilidade de defasagem (defasagem < 0) com modelo treinado."""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pede_cleaning import PROJECT_ROOT
from src.pede_model import default_model_path, load_bundle, predict_row
from src.streamlit_bootstrap import ensure_streamlit_artifacts

ensure_streamlit_artifacts(PROJECT_ROOT)

FEATURE_HELP: dict[str, str] = {
    "idade_referencia": (
        "**O que é:** idade do aluno (ou idade base) usada na linha harmonizada para o painel.\n\n"
        "**Como usar:** escolha entre as idades que realmente aparecem na base."
    ),
    "ano_ingresso": (
        "**O que é:** ano em que o aluno entrou na instituição (Passos).\n\n"
        "**Como usar:** apenas anos presentes nos dados deste projeto."
    ),
    "ano_cohorte": (
        "**O que é:** **ano do painel** em que aquela linha foi observada (no banco: coluna `ano_cohorte`, valores 2022, 2023 ou 2024). "
        "É o ano da “foto” do PEDE na base harmonizada — não confundir com **ano de ingresso**.\n\n"
        "**Importância:** contextualiza mudanças de formulário e de política por ano."
    ),
    "cg": (
        "**O que é:** contador/acúmulo de marcas relativas ao eixo CG no acompanhamento pedagógico (convenção dos CSVs).\n\n"
        "**Como usar:** dígitos inteiros dentro da faixa observada na base para evitar extrapolação."
    ),
    "cf": "**O que é:** análogo ao CG, mas para o eixo **CF**. Valores são contínuos/discretos conforme construção da base.",
    "ct": (
        "**O que é:** nível ou marca do CT (caracterização técnica do PEDE).\n\n"
        "**Como usar:** escolha entre os valores inteiros que apareceram nos dados."
    ),
    "n_avaliacoes": (
        "**O que é:** quantas avaliações pedagógicas (e componentes) entram nesta linha (0 até o máximo típico).\n\n"
        "**Leitura:** mais avaliações podem correlacionar com mais dados de gestão sobre o aluno."
    ),
    "iaa": (
        "**O que é:** Indicador de Aprendizagem Ajustado (componente Pedagógica).\n\n"
        "**Escala:** próximo de faixa 0–10 em boa parte dos relatórios; use a faixa exibida abaixo."
    ),
    "ieg": "**O que é:** Indicador de Engajamento — hábitos, participação e comprometimento com as atividades escolares.",
    "ips": "**O que é:** Indicador Psicossocial — bem-estar, vínculos e tensões nas dimensões acompanhadas pela Passos.",
    "ipp": "**O que é:** Indicador voltado ao **papel/desempenho** em situações de aprendizado e comportamento esperado.",
    "ida": "**O que é:** Indicador de Desempenho Acadêmico — resultados objetivos nos eixos letrados/disciplinas.",
    "mat": "**O que é:** nota ou componente relacionado ao eixo **Matemática** (conforme coluna na base unificada).",
    "por": "**O que é:** mesmo para **Português**.",
    "ing": "**O que é:** mesmo para **Inglês**.",
    "ipv": "**O que é:** Indicador de vulnerabilidade em dimensões acompanhadas pelo psicopedagógico (escala habitual 2,5–10).",
    "inde_hist_22": (
        "**O que é:** valor do **INDE** (índice composto institucional) associado ao histórico na janela **2022**, "
        "derivado ao preparar a base para ML — **não** é obrigatoriamente o INDE atual da mesma linha da pergunta 1–10.\n\n"
        "**Como usar:** ajuste com **passo de décimos** (+/−); incrementos muito pequenos foram evitados para o controle +/- funcionar bem."
    ),
    "inde_hist_23": (
        "**O que é:** como o campo anterior, para a **janela 2023**.\n\n"
        "Serve ao modelo como memória dos anos anteriores do aluno no painel, quando disponível nos dados brutos."
    ),
    "genero": "**O que é:** gênero registrado nos dados sociodemográficos (categorias exatamente como aparecem no CSV).",
}


def _label(col: str) -> str:
    if col == "genero":
        return "Gênero"
    if col == "inde_hist_22":
        return "INDE histórico (2022)"
    if col == "inde_hist_23":
        return "INDE histórico (2023)"
    return {
        "idade_referencia": "Idade (referência)",
        "ano_ingresso": "Ano de ingresso",
        "ano_cohorte": "Ano do painel PEDE",
        "cg": "CG",
        "cf": "CF",
        "ct": "CT",
        "n_avaliacoes": "Nº de avaliações",
        "iaa": "IAA",
        "ieg": "IEG",
        "ips": "IPS",
        "ipp": "IPP",
        "ida": "IDA",
        "mat": "Matemática",
        "por": "Português",
        "ing": "Inglês",
        "ipv": "IPV",
    }.get(col, col.replace("_", " ").title())


def _obs_bounds(df: pd.DataFrame | None, col: str) -> tuple[float, float]:
    if df is None or col not in df.columns:
        return float("-inf"), float("inf")
    s = df[col].dropna()
    if s.empty:
        return float("-inf"), float("inf")
    return float(s.min()), float(s.max())


def _snap_to_step(lo: float, hi: float, v: float, step: float) -> float:
    """Alinha o valor à grade do número de passos (evita problemas nos botões +/- do Streamlit)."""
    if step <= 0:
        return float(min(max(v, lo), hi))
    n = round((v - lo) / step)
    snapped = lo + n * step
    return float(min(max(snapped, lo), hi))


def _padded_bounds(lo: float, hi: float, *, pad_ratio: float = 0.08) -> tuple[float, float]:
    if not math.isfinite(lo) or not math.isfinite(hi):
        return 0.0, 1.0
    if lo >= hi:
        return lo, lo + 1.0
    span = hi - lo
    pad = max(span * pad_ratio, 1e-4)
    return lo - pad, hi + pad


def _int_uniques(df: pd.DataFrame | None, col: str) -> list[int]:
    if df is None or col not in df.columns:
        return []
    s = df[col].dropna()
    out = sorted({int(round(float(x))) for x in s.unique()})
    return out


def _nearest_index(options: list[int], target: float) -> int:
    if not options:
        return 0
    t = int(round(target))
    if t in options:
        return options.index(t)
    return min(range(len(options)), key=lambda i: abs(options[i] - t))


def _collect_out_of_range(
    df: pd.DataFrame | None,
    row: dict[str, float | str],
    numeric_cols: list[str],
) -> list[str]:
    if df is None:
        return []
    msgs: list[str] = []
    for c in numeric_cols:
        lo, hi = _obs_bounds(df, c)
        if not math.isfinite(lo) or not math.isfinite(hi):
            continue
        v = float(row[c])
        if v < lo - 1e-9 or v > hi + 1e-9:
            msgs.append(
                f"**{_label(c)}:** valor informado **{v:g}** está **fora** da faixa observada na base de treino "
                f"(**{lo:g}** a **{hi:g}**). O modelo pode extrapolar; interprete com cautela."
            )
    return msgs


MODEL_PATH = default_model_path(PROJECT_ROOT)
PARQUET = PROJECT_ROOT / "data_processed" / "pede_unificado.parquet"

bundle = load_bundle(MODEL_PATH)
df_ref = pd.read_parquet(PARQUET) if PARQUET.exists() else None

defaults_num = bundle["defaults_numeric"]
defaults_cat = bundle["defaults_cat"]
num_cols = bundle["numeric_features"]

head_l, head_r = st.columns([4, 1])
with head_l:
    st.title("Modelo — risco de defasagem")
with head_r:
    if st.button("Voltar aos Insights", use_container_width=True):
        st.switch_page("pages/2_Insights.py")

st.caption(
    "Escolha **anos e idades** em listas (sem decimais estranhos). Demais campos: **número** com limites amplos; "
    "se estiver fora do que a base já viu, mostramos **avisos** ao estimar."
)

st.markdown(
    "<sub> Nos campos com o ícone **?**, o texto explica cada variável e mostra **a faixa já observada na base** neste projeto. "
    "Digitar fora dessa faixa é permitido, mas valores fora disso disparam um aviso ao clicar em **Estimar risco** (extrapolação). "
    "</sub>",
    unsafe_allow_html=True,
)

with st.expander("Definição do alvo, features e limitações", expanded=False):
    st.markdown(
        bundle.get(
            "target_definition",
            "Alvo: indicador binário **defasagem &lt; 0** (fase efetiva abaixo da ideal).",
        )
    )
    st.markdown(
        "- **Entradas:** indicadores psicoacadêmicos e histórico de INDE **sem** `ian` nem `defasagem`.\n"
        "- **Uso:** apoio à triagem; não substitui avaliação humana.\n"
        "- **Retreino:** `python scripts/train_model.py` após atualizar o parquet."
    )

with st.expander("Métricas no holdout (bundle atual)", expanded=False):
    st.text(bundle["metrics"].get("classification_report", ""))
    st.metric("ROC-AUC (teste)", f"{bundle['metrics'].get('roc_auc', float('nan')):.3f}")

row: dict[str, float | str] = {}

_shell = st.container(border=True) if "border" in inspect.signature(st.container).parameters else st.container()

with _shell:
    st.markdown("##### Cenário para predição")
    st.caption("Anos e contagens discretas em **listas**; indicadores contínuos em **campo numérico**.")

    _tab_ctx, _tab_ped, _tab_inde = st.tabs(["Contexto do aluno", "Indicadores PEDE", "INDE Histórico"])

    with _tab_ctx:
        c1, c2, c3 = st.columns(3)
        anos_cohorte = _int_uniques(df_ref, "ano_cohorte") or [2022, 2023, 2024]
        anos_ing = _int_uniques(df_ref, "ano_ingresso") or list(range(2016, 2025))
        idades = _int_uniques(df_ref, "idade_referencia") or list(range(7, 28))

        with c1:
            row["ano_cohorte"] = float(
                st.selectbox(
                    _label("ano_cohorte"),
                    options=anos_cohorte,
                    format_func=lambda y: str(int(y)),
                    index=_nearest_index(anos_cohorte, defaults_num.get("ano_cohorte", anos_cohorte[0])),
                    help=FEATURE_HELP["ano_cohorte"],
                )
            )
        with c2:
            row["ano_ingresso"] = float(
                st.selectbox(
                    _label("ano_ingresso"),
                    options=anos_ing,
                    format_func=lambda y: str(int(y)),
                    index=_nearest_index(anos_ing, defaults_num.get("ano_ingresso", anos_ing[0])),
                    help=FEATURE_HELP["ano_ingresso"],
                )
            )
        with c3:
            row["idade_referencia"] = float(
                st.selectbox(
                    _label("idade_referencia"),
                    options=idades,
                    format_func=lambda y: str(int(y)),
                    index=_nearest_index(idades, defaults_num.get("idade_referencia", idades[len(idades) // 2])),
                    help=FEATURE_HELP["idade_referencia"],
                )
            )

        opts_gen = (
            sorted(df_ref["genero"].dropna().astype(str).unique().tolist())
            if df_ref is not None
            else [str(defaults_cat.get("genero", "F"))]
        )
        dv = str(defaults_cat.get("genero", opts_gen[0]))
        idx_g = opts_gen.index(dv) if dv in opts_gen else 0
        row["genero"] = st.selectbox(
            _label("genero"),
            opts_gen,
            index=idx_g,
            help=FEATURE_HELP["genero"],
        )

    def _num_field(
        col: str,
        *,
        step: float | None,
        fmt: str | None = None,
        step_mode: str = "default",
    ) -> None:
        lo_o, hi_o = _obs_bounds(df_ref, col)
        d = float(defaults_num.get(col, (lo_o + hi_o) / 2 if math.isfinite(lo_o) else 0.0))
        if not math.isfinite(lo_o):
            lo_o, hi_o = d - 1.0, d + 1.0
        lo_w, hi_w = _padded_bounds(lo_o, hi_o)
        d = min(max(d, lo_w), hi_w)
        # INDE histórico: passo muito fino (0.001) quebra +/- no widget; usar décimos + grade alinhada
        effective_step = step
        effective_fmt = fmt
        if step_mode == "inde":
            span = hi_w - lo_w
            effective_step = 0.01 if span <= 25 else min(0.05, span / 200.0)
            effective_fmt = "%.2f"
            d = _snap_to_step(lo_w, hi_w, d, effective_step)
        kwargs: dict = {
            "label": _label(col),
            "min_value": float(lo_w),
            "max_value": float(hi_w),
            "value": float(d),
            "help": (
                (FEATURE_HELP.get(col, "") or "").strip()
                + "\n\n---\n\n**Faixa já observada na base (treino):** "
                f"**{lo_o:g}** a **{hi_o:g}**."
                "\n\nValores fora dessa faixa geram um aviso ao clicar em **Estimar risco**."
            ),
        }
        if effective_step is not None:
            kwargs["step"] = float(effective_step)
        if effective_fmt is not None:
            kwargs["format"] = effective_fmt
        kwargs["key"] = f"ni_{col}_snap01" if step_mode == "inde" else f"ni_{col}"
        row[col] = float(st.number_input(**kwargs))

    with _tab_ped:
        st.markdown("**Acompanhamento (contagens / marcadores)**")
        p1, p2 = st.columns(2)
        n_aval_opts = _int_uniques(df_ref, "n_avaliacoes") or [0, 1, 2, 3, 4, 5, 6]
        ct_opts = _int_uniques(df_ref, "ct") or list(range(1, 19))

        with p1:
            row["n_avaliacoes"] = float(
                st.selectbox(
                    _label("n_avaliacoes"),
                    options=n_aval_opts,
                    format_func=lambda x: str(int(x)),
                    index=_nearest_index(n_aval_opts, defaults_num.get("n_avaliacoes", 0)),
                    help=FEATURE_HELP["n_avaliacoes"],
                )
            )
        with p2:
            row["ct"] = float(
                st.selectbox(
                    _label("ct"),
                    options=ct_opts,
                    format_func=lambda x: str(int(x)),
                    index=_nearest_index(ct_opts, defaults_num.get("ct", ct_opts[0])),
                    help=FEATURE_HELP["ct"],
                )
            )
        p3, p4 = st.columns(2)
        with p3:
            _num_field("cg", step=1.0, fmt="%.0f")
        with p4:
            _num_field("cf", step=1.0, fmt="%.0f")

        st.markdown("**Indicadores (0–10 e IPV)**")
        g1, g2, g3 = st.columns(3)
        with g1:
            _num_field("iaa", step=0.1)
            _num_field("ieg", step=0.1)
            _num_field("ips", step=0.1)
        with g2:
            _num_field("ipp", step=0.1)
            _num_field("ida", step=0.1)
            _num_field("ipv", step=0.05)
        with g3:
            _num_field("mat", step=0.1)
            _num_field("por", step=0.1)
            _num_field("ing", step=0.1)

    with _tab_inde:
        st.markdown("**INDE Histórico (features do modelo)**")
        i1, i2 = st.columns(2)
        with i1:
            _num_field("inde_hist_22", step=0.01, step_mode="inde")
        with i2:
            _num_field("inde_hist_23", step=0.01, step_mode="inde")

st.divider()
c_btn, _ = st.columns([1, 2])
with c_btn:
    run = st.button("Estimar risco", type="primary", use_container_width=True)

if run:
    avisos = _collect_out_of_range(df_ref, row, num_cols)
    for a in avisos:
        st.warning(a)

    p = predict_row(bundle, row)
    out = st.container(border=True) if "border" in inspect.signature(st.container).parameters else st.container()
    with out:
        st.markdown("##### Resultado")
        m1, m2 = st.columns([1, 2])
        with m1:
            st.metric("P(defasagem < 0)", f"{p:.1%}")
        with m2:
            st.caption("Intensidade do risco (0–100%)")
            st.progress(min(max(p, 0.0), 1.0))
        if p >= 0.55:
            st.warning(
                "**Alerta:** probabilidade elevada — priorize revisão do plano pedagógico e do acompanhamento psicossocial."
            )
        elif p <= 0.35:
            st.success("Probabilidade **baixa** no modelo (sempre validar com o caso concreto na equipe).")
        else:
            st.info("**Zona intermediária** — combine com outros sinais (fase, frequência, entrevista).")
