"""
Gráficos e **respostas** às perguntas 1–10 do case (EDA sobre `pede_unificado.parquet`).

Observação metodológica global: os arquivos são **recortes por ano do painel** (coluna `ano_cohorte`: 2022, 2023, 2024);
não há medidas mensais “ao longo do ano” no CSV. Onde o enunciado fala em evolução temporal, usamos **2022→2024**
e/ou **série por RA** quando o painel permite.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pede_cleaning import PROJECT_ROOT


def _ian_categoria(ian: pd.Series) -> pd.Series:
    """Faixas alinhadas ao material PEDE (IAN 10 / 5 / 2,5)."""
    s = pd.to_numeric(ian, errors="coerce")
    out: list[str] = []
    for v in s:
        if pd.isna(v):
            out.append("Sem IAN")
        elif v >= 9.0:
            out.append("Em fase (IAN 10)")
        elif v >= 4.0:
            out.append("Moderada (IAN 5)")
        else:
            out.append("Severa (IAN 2,5)")
    return pd.Series(out, index=s.index, dtype=object)


def _pedra_ordem(s: pd.Series) -> pd.Series:
    def _one(t: str) -> float:
        u = str(t).lower()
        if "quartzo" in u:
            return 1.0
        if "ágata" in u or "agata" in u:
            return 2.0
        if "ametista" in u:
            return 3.0
        if "topázio" in u or "topazio" in u:
            return 4.0
        return np.nan

    return s.map(_one)


def tab_q1_ian(df: pd.DataFrame) -> tuple[Any, str]:
    d = df.copy()
    d["ian_cat"] = _ian_categoria(d["ian"])
    cnt = d.groupby(["ano_cohorte", "ian_cat"]).size().reset_index(name="n")
    fig = px.bar(
        cnt,
        x="ano_cohorte",
        y="n",
        color="ian_cat",
        barmode="group",
        title="Perfil de IAN por ano do painel (em fase / moderada / severa)",
        labels={"n": "Nº de registros", "ano_cohorte": "Ano"},
    )
    n = len(d)
    mod = int((d["ian_cat"] == "Moderada (IAN 5)").sum())
    sev = int((d["ian_cat"] == "Severa (IAN 2,5)").sum())
    emf = int((d["ian_cat"] == "Em fase (IAN 10)").sum())
    txt = f"""
#### Resposta à pergunta 1

- **Perfil geral:** entre todos os registros ({n:,}), **{emf:,}** estão **em fase (IAN 10)**,
  **{mod:,}** em **defasagem moderada (IAN 5)** e **{sev:,}** em **severa (IAN 2,5)** — ou seja,
  **{mod + sev:,}** alunos-registros (**{(mod + sev) / n:.1%}**) em moderada ou severa.
- **“Ao longo do ano”:** o CSV **não traz meses/bimestres**; a evolução observável aqui é **entre anos do painel**
  (barras por 2022, 2023 e 2024). Compare as barras de moderada/severa para ver se o recorte piora ou melhora no tempo.
- **Implicação:** concentre ações de reforço de nível onde moderada+severa crescer no último ano disponível.
""".strip()
    return fig, txt


def tab_q2_ida(df: pd.DataFrame) -> tuple[Any, str]:
    d = df.dropna(subset=["ida", "fase"])
    agg = d.groupby(["ano_cohorte", "fase"], as_index=False)["ida"].mean()
    agg = agg.sort_values(["ano_cohorte", "fase"])
    fig = px.line(
        agg,
        x="fase",
        y="ida",
        color="ano_cohorte",
        markers=True,
        title="IDA médio por fase e por ano do painel",
        labels={"ida": "IDA médio", "fase": "Fase"},
    )
    by_year = d.groupby("ano_cohorte")["ida"].mean().sort_index()
    if len(by_year) < 2:
        sent = "Há apenas um ano na base filtrada; não dá para falar em tendência temporal."
    else:
        a0, a1 = float(by_year.iloc[0]), float(by_year.iloc[-1])
        if abs(a1 - a0) < 0.15:
            sent = f"O IDA médio global oscila pouco (**{a0:.2f}** → **{a1:.2f}**): tendência **estagnada** entre o primeiro e o último ano."
        elif a1 > a0:
            sent = f"O IDA médio global **melhora** de **{a0:.2f}** para **{a1:.2f}** entre o primeiro e o último ano."
        else:
            sent = f"O IDA médio global **cai** de **{a0:.2f}** para **{a1:.2f}** entre o primeiro e o último ano."
    anos = ", ".join(f"**{int(y)}**: {v:.2f}" for y, v in by_year.items())
    txt = f"""
#### Resposta à pergunta 2

- **Ao longo dos anos (média global):** {anos}. {sent}
- **Ao longo das fases:** use o gráfico: curvas mais altas indicam melhor IDA médio naquela fase para aquele ano.
  Fases com curva descendente merecem revisão de conteúdo e carga horária de apoio.
- **Implicação:** combine “tendência entre anos” com “fases mais frágeis” para priorizar intervenções.
""".strip()
    return fig, txt


def tab_q3_ieg_ida_ipv(df: pd.DataFrame) -> tuple[Any, str]:
    d = df.dropna(subset=["ieg", "ida", "ipv"])
    fig = px.scatter(
        d,
        x="ieg",
        y="ida",
        color="ipv",
        facet_col="ano_cohorte",
        title="IEG × IDA (cor contínua = IPV)",
        labels={"ieg": "IEG", "ida": "IDA", "ipv": "IPV"},
        height=500,
    )
    r_ie_ida = d["ieg"].corr(d["ida"])
    r_ie_ipv = d["ieg"].corr(d["ipv"])
    r_id_ipv = d["ida"].corr(d["ipv"])
    txt = f"""
#### Resposta à pergunta 3

- **IEG e IDA:** correlação de Pearson **≈ {r_ie_ida:.2f}** — há **associação positiva** (maior engajamento tende a
  acompanhar melhor desempenho acadêmico), o que responde “sim” a uma **relação direta** no sentido estatístico
  (não implica causalidade isolada).
- **IEG e IPV:** **≈ {r_ie_ipv:.2f}**; **IDA e IPV:** **≈ {r_id_ipv:.2f}**. O ponto de virada **co-varia** com desempenho
  e, em menor grau, com o padrão de engajamento neste recorte.
- **Implicação:** estratégias que aumentem IEG e IDA são coerentes com elevar IPV; avalie casos dispersos (IEG alto e IDA baixo) na equipe pedagógica.
""".strip()
    return fig, txt


def tab_q4_iaa(df: pd.DataFrame) -> tuple[Any, str]:
    d = df.dropna(subset=["iaa", "ida", "ieg"])
    fig1 = px.scatter(
        d.sample(min(2000, len(d)), random_state=42),
        x="ida",
        y="iaa",
        color="ano_cohorte",
        title="IAA × IDA (cada ponto é um aluno-ano)",
        labels={"ida": "IDA", "iaa": "IAA"},
    )
    r_iaa_ida = d["iaa"].corr(d["ida"])
    r_iaa_ieg = d["iaa"].corr(d["ieg"])
    dispersos = int(((d["iaa"] >= 8) & (d["ida"] < 5)).sum())
    leitura_ida = (
        "baixa" if abs(r_iaa_ida) < 0.25 else "moderada" if abs(r_iaa_ida) < 0.55 else "alta"
    )
    leitura_ieg = (
        "baixa" if abs(r_iaa_ieg) < 0.25 else "moderada" if abs(r_iaa_ieg) < 0.55 else "alta"
    )
    txt = f"""
#### Resposta à pergunta 4

- **Coerência IAA × IDA:** correlação **≈ {r_iaa_ida:.2f}** — coerência agregada **{leitura_ida}**.
  O IAA vem de autoavaliação padronizada, então mede percepção/escuta do aluno, não desempenho direto.
- **Coerência IAA × IEG:** **≈ {r_iaa_ieg:.2f}** — coerência agregada **{leitura_ieg}**.
- **Incoerências úteis:** existem **{dispersos:,}** registros com **IAA alto (≥8)** e **IDA baixo (<5)** —
  percepção melhor que o resultado; merecem conversa e diagnóstico (dificuldade técnica, teste, frequência etc.).
- **Implicação:** use IAA como escuta ativa, não como substituto de IDA/IEG.
""".strip()
    return fig1, txt


def tab_q5_ips(df: pd.DataFrame) -> tuple[Any, str]:
    d = df.sort_values(["ra", "ano_cohorte"]).copy()
    d["ida_next"] = d.groupby("ra")["ida"].shift(-1)
    d["ieg_next"] = d.groupby("ra")["ieg"].shift(-1)
    d["delta_ida"] = d["ida_next"] - d["ida"]
    d["delta_ieg"] = d["ieg_next"] - d["ieg"]
    sub = d.dropna(subset=["ips", "delta_ida"]).copy()
    sub["queda_ida"] = (sub["delta_ida"] < -0.5).astype(int)
    fig = px.box(
        sub,
        x="queda_ida",
        y="ips",
        color="ano_cohorte",
        title="IPS no ano t vs queda forte de IDA no ano seguinte (painel por RA)",
        labels={"queda_ida": "0 = sem queda; 1 = queda IDA >0,5", "ips": "IPS"},
    )
    m0 = sub.loc[sub["queda_ida"] == 0, "ips"].median()
    m1 = sub.loc[sub["queda_ida"] == 1, "ips"].median()
    sub2 = d.dropna(subset=["ips", "delta_ieg"]).copy()
    sub2["queda_ieg"] = (sub2["delta_ieg"] < -0.5).astype(int)
    m0e = sub2.loc[sub2["queda_ieg"] == 0, "ips"].median()
    m1e = sub2.loc[sub2["queda_ieg"] == 1, "ips"].median()
    txt = f"""
#### Resposta à pergunta 5

- **Desempenho acadêmico (IDA):** mediana de IPS **sem** queda forte de IDA no ano seguinte: **{m0:.2f}**;
  **com** queda: **{m1:.2f}**. Quem depois cai em IDA tende a ter IPS ligeiramente mais baixo no ano anterior
  (associação; **não** prova causalidade).
- **Engajamento (IEG):** mediana de IPS sem queda forte de IEG no ano seguinte: **{m0e:.2f}**; com queda: **{m1e:.2f}**.
- **Implicação:** IPS baixo + tendência de queda em IDA/IEG no painel funciona como **alerta combinado** para psicologia e pedagógico.
""".strip()
    return fig, txt


def tab_q6_ipp_ian(df: pd.DataFrame) -> tuple[Any, str]:
    d_all = df.dropna(subset=["ian"])
    d = df.dropna(subset=["ipp", "ian"])
    if len(d) < 30:
        fig = go.Figure()
        fig.add_annotation(text="Poucos registros com IPP e IAN válidos", showarrow=False)
        txt = """
#### Resposta à pergunta 6

- **Dado:** no layout **2022** não há coluna IPP no CSV original; a maior parte das linhas com IPP vem de **2023/2024**.
- **Regra PEDE:** no INDE oficial, **IPP é ponderado apenas nas fases 0–7**; para **fase 8** ele é **N/A**.
- **Conclusão:** com amostra pequena, não é possível afirmar confirmação/contradição robusta entre IPP e IAN neste app.
- **Implicação:** replique o cruzamento no notebook com testes e exemplos de caso a caso.
""".strip()
        return fig, txt
    fig = px.scatter(
        d.sample(min(2500, len(d)), random_state=1),
        x="ian",
        y="ipp",
        color="ano_cohorte",
        title="IPP × IAN (IAN alto = menor defasagem no critério PEDE)",
        labels={"ian": "IAN", "ipp": "IPP"},
    )
    r = d["ian"].corr(d["ipp"])
    if r >= 0.25:
        conc = "**confirmam** em média: IPP mais alto aparece junto de IAN mais alto (menos defasagem)."
    elif r <= -0.25:
        conc = "sugerem **contradição** agregada: IPP alto associado a IAN mais baixo — investigar casos e qualidade dos dados."
    else:
        conc = "mostram **associação fraca** entre IPP e IAN neste recorte; a confirmação/contradição deve ser feita **por turma/caso**."
    txt = f"""
#### Resposta à pergunta 6

- **Leitura agregada:** correlação Pearson **IAN × IPP ≈ {r:.2f}** em **{len(d):,}** registros com ambos preenchidos
  (de {len(d_all):,} com IAN).
- **Regra PEDE:** no INDE oficial, **IPP é ponderado apenas nas fases 0–7**; para **fase 8** ele é **N/A**.
- **Confirmam ou contradizem?** No agregado, os dados {conc}
- **Implicação:** em divergências pontuais (IAN severo e IPP alto), marque revisão multidisciplinar.
""".strip()
    return fig, txt


def tab_q7_ipv(df: pd.DataFrame) -> tuple[Any, str]:
    d = df.dropna(subset=["ipv", "ida", "ieg", "iaa", "ips"])
    cols = ["ida", "ieg", "iaa", "ips", "mat", "por", "ing"]
    corr = d[["ipv"] + cols].corr()["ipv"].drop("ipv").sort_values(key=abs, ascending=False)
    cdf = pd.DataFrame({"variavel": corr.index.astype(str), "r": corr.values})
    fig = px.bar(cdf, x="variavel", y="r", title="Correlação com IPV (proxy de “comportamentos” medidos pelos indicadores)")
    mapeamento = {
        "ida": "acadêmico (desempenho agregado)",
        "mat": "acadêmico (Matemática)",
        "por": "acadêmico (Português)",
        "ing": "acadêmico (Inglês)",
        "ieg": "engajamento",
        "iaa": "autoimagem / percepção (proxy emocional)",
        "ips": "psicossocial",
    }
    linhas = []
    for nome, r in corr.head(5).items():
        linhas.append(f"- **{nome}** ({mapeamento.get(nome, 'indicador')}), r = **{r:+.2f}**")
    corpo = "\n".join(linhas)
    txt = f"""
#### Resposta à pergunta 7

- **“Ao longo do tempo”:** aqui usamos **correlação no painel aluno-ano** como proxy; indicadores com maior |r| com IPV
  são os que mais **co-movem** com o ponto de virada neste recorte.
- **Principais associações:**
{corpo}
- **Implicação:** IPV sobe junto de desempenho acadêmico e, em segundo plano, de IEG/IAA/IPS — intervenções isoladas raramente explicam o ponto de virada.
""".strip()
    return fig, txt


def tab_q8_inde_dim(df: pd.DataFrame) -> tuple[Any, str]:
    oficiais_0_7 = ["ian", "ida", "ieg", "iaa", "ips", "ipp", "ipv"]
    d = df.dropna(subset=["inde_cohorte", "fase"]).copy()
    d = d[d["fase"] < 8]
    d = d.dropna(subset=[c for c in oficiais_0_7 if c in d.columns])
    corr = d[["inde_cohorte"] + oficiais_0_7].corr()["inde_cohorte"].drop("inde_cohorte").sort_values(ascending=False)
    cdf = pd.DataFrame({"indicador": corr.index.astype(str), "r": corr.values})
    fig = px.bar(cdf, x="indicador", y="r", title="Correlação de cada dimensão oficial com o INDE (fases 0–7)")
    melhor = corr.index[0]
    pesos_0_7 = "IAN 10%, IDA 20%, IEG 20%, IAA 10%, IPS 10%, IPP 10%, IPV 20%"
    pesos_8 = "IAN 10%, IDA 40%, IEG 20%, IAA 10%, IPS 20% (IPP/IPV N/A)"
    txt = f"""
#### Resposta à pergunta 8

- **Regra oficial do INDE:** fases **0–7** usam **{pesos_0_7}**; fase **8** usa **{pesos_8}**.
- **Combinação nas fases 0–7:** as dimensões oficiais se associam positivamente ao **INDE**; a maior correlação simples
  neste painel é **{melhor}** (r = **{corr.iloc[0]:.2f}**).
- **Leitura:** pela fórmula, **IDA, IEG e IPV** têm maior peso nas fases 0–7; na fase 8, **IDA** ganha peso central.
  As correlações do gráfico são diagnóstico exploratório e **não substituem** a fórmula ponderada oficial.
- **Implicação:** priorize combinações que preservem equilíbrio, mas dê atenção extra às dimensões com maior peso oficial em cada fase.
""".strip()
    return fig, txt


def tab_q9_ml(df: pd.DataFrame) -> tuple[Any, str]:
    """Histograma de P(risco) na base + importâncias do Random Forest (se o bundle carregar)."""
    from src.pede_model import build_xy

    model_path = PROJECT_ROOT / "App" / "models" / "risk_defasagem.joblib"
    auc_txt = ""
    feats = "idade, notas componentes, IAA, IEG, IPS, IPP, IDA, IPV, histórico de INDE, ano do painel (`ano_cohorte`), gênero."
    chart_note = ""

    fig = go.Figure()
    fig.add_annotation(
        text="Treine o modelo: python scripts/train_model.py",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14),
    )
    fig.update_layout(title="Pergunta 9 — modelo", height=400)

    if model_path.exists():
        try:
            import joblib

            bundle = joblib.load(model_path)
            pipe = bundle["pipeline"]
            auc = bundle.get("metrics", {}).get("roc_auc")
            if auc is not None:
                auc_txt = f" No holdout, **ROC-AUC = {auc:.2f}**."
            fn = bundle.get("numeric_features", [])
            fc = bundle.get("categorical_features", [])
            feats = ", ".join(fn + fc)

            X, _ = build_xy(df)
            proba = pipe.predict_proba(X)[:, 1]

            prep = pipe.named_steps["prep"]
            clf = pipe.named_steps["clf"]
            names = prep.get_feature_names_out()
            imp = clf.feature_importances_
            imp_df = (
                pd.DataFrame({"feature": names, "importance": imp})
                .sort_values("importance", ascending=False)
                .head(18)
            )

            fig = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.42, 0.58],
                vertical_spacing=0.12,
                subplot_titles=(
                    "Distribuição de P(risco) em todos os registros (modelo aplicado à base)",
                    "Variáveis mais usadas pelo Random Forest (após pré-processamento)",
                ),
            )
            fig.add_trace(
                go.Histogram(x=proba, nbinsx=35, name="P(risco)", marker_color="#636EFA"),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Bar(
                    x=imp_df["importance"],
                    y=imp_df["feature"],
                    orientation="h",
                    name="importância",
                    marker_color="#00CC96",
                ),
                row=2,
                col=1,
            )
            fig.update_xaxes(title_text="Probabilidade P(risco)", row=1, col=1)
            fig.update_xaxes(title_text="Importância", row=2, col=1)
            fig.update_yaxes(automargin=True)
            fig.update_layout(height=820, showlegend=False, title_text="Pergunta 9 — padrões do modelo na base atual")
            chart_note = (
                " O gráfico superior mostra **quantos alunos-registros** ficam em cada faixa de risco quando aplicamos "
                "o modelo a **toda** a base (para visão populacional, não para validação). O inferior resume **quais "
                "combinações de features transformadas** o floresta mais usa."
            )
        except Exception as exc:  # noqa: BLE001
            auc_txt = f" Não foi possível carregar ou executar o modelo (`{exc!s}`). Regrave o joblib com o **mesmo** scikit-learn do `requirements.txt`."
    else:
        auc_txt = " Arquivo do modelo ausente — rode `python scripts/train_model.py`."

    txt = f"""
#### Resposta à pergunta 9

- **Padrões:** o **Random Forest** usa **{feats}** para estimar **P(defasagem &lt; 0)** (fase abaixo da ideal), **sem**
  colocar IAN/defasagem nas entradas.{auc_txt}{chart_note}
- **Simulador:** ajuste valores manualmente na página **Modelo / Predição** (veja o botão abaixo do gráfico nesta aba).
- **Implicação:** triagem estatística; decisões continuam com a equipe pedagógica/psicossocial.
""".strip()
    return fig, txt


def tab_q10_efetividade(df: pd.DataFrame) -> tuple[Any, str]:
    d = df.copy()
    d["p_ord"] = _pedra_ordem(d["pedra_atual"])
    agg = d.dropna(subset=["p_ord"]).groupby("ano_cohorte")["p_ord"].mean()
    fig = px.bar(
        x=agg.index.astype(int),
        y=agg.values,
        title="Pedra atual — ordem média (1=Quartzo … 4=Topázio) por ano do painel",
        labels={"x": "Ano", "y": "Ordem média"},
    )
    if len(agg) >= 2:
        t0, t1 = float(agg.iloc[0]), float(agg.iloc[-1])
        if t1 > t0 + 0.05:
            prog = "há **melhora agregada** na pedra média entre o primeiro e o último ano (ordem mais alta = pedra “melhor”)."
        elif t1 < t0 - 0.05:
            prog = "a pedra média **cai** entre o primeiro e o último ano — **não** evidencia melhora consistente nesse proxy."
        else:
            prog = "a pedra média permanece **estável** entre o primeiro e o último ano."
    else:
        prog = "há **apenas um ano**; não dá para concluir tendência de efetividade."

    share_top = (
        d.assign(is_top=d["pedra_atual"].astype(str).str.contains("topázio|topazio", case=False, regex=True))
        .groupby("ano_cohorte")["is_top"]
        .mean()
    )

    txt = f"""
#### Resposta à pergunta 10

- **Proxy de “melhora no ciclo”:** as **Pedras** são faixas/conceitos derivados do INDE, não fases escolares.
  Ordenamos Quartzo→Ágata→Ametista→Topázio e calculamos a **média por ano** da pedra atual.
  {prog.capitalize()}
- **Topázio (fração de alunos):** {", ".join(f"**{int(y)}**: {float(v):.1%}" for y, v in share_top.items())} — ajuda a ver se o recorte **premium** cresce.
- **Ressalva:** isso **não prova impacto causal** do programa (seleção, entrada/saída de alunos, mudança de regra). Confirmação de impacto exige desenho contrafactual ou séries mais longas.
- **Implicação:** use junto com IDA/INDE e permanência para narrar resultados a stakeholders.
""".strip()
    return fig, txt


TAB_FUNCS = [
    tab_q1_ian,
    tab_q2_ida,
    tab_q3_ieg_ida_ipv,
    tab_q4_iaa,
    tab_q5_ips,
    tab_q6_ipp_ian,
    tab_q7_ipv,
    tab_q8_inde_dim,
    tab_q9_ml,
    tab_q10_efetividade,
]

TAB_TITLES = [
    "1 — IAN",
    "2 — IDA",
    "3 — IEG",
    "4 — IAA",
    "5 — IPS",
    "6 — IPP",
    "7 — IPV",
    "8 — INDE",
    "9 — ML",
    "10 — Efetividade",
]
