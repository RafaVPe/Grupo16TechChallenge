"""
Leituras automáticas: textos derivados dos dados (painel unificado) e do recorte filtrado.

Compara o subconjunto filtrado à base completa quando o filtro não cobre 100% dos registros.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def aplicar_filtros(
    df: pd.DataFrame,
    anos: list[int] | None,
    generos: list[str] | None,
    fases: list[str] | None,
) -> pd.DataFrame:
    out = df.copy()
    if anos:
        out = out[out["ano_cohorte"].isin(anos)]
    if generos:
        out = out[out["genero"].isin(generos)]
    if fases:
        out = out[out["fase"].astype(str).isin(fases)]
    return out


def _painel_stats(full: pd.DataFrame) -> dict[str, Any]:
    """Estatísticas de referência calculadas na base completa (sempre atualizadas)."""
    n = len(full)
    neg = full.assign(_n=full["defasagem"] < 0).groupby("ano_cohorte", observed=True)["_n"].mean()
    inde_m = full.groupby("ano_cohorte", observed=True)["inde_cohorte"].median()
    ieg_m = full.groupby("ano_cohorte", observed=True)["ieg"].median()
    ida_m = full.groupby("ano_cohorte", observed=True)["ida"].median()
    ips_m = full.groupby("ano_cohorte", observed=True)["ips"].median()
    t_global = float((full["defasagem"] < 0).mean())
    anos_ord = sorted(full["ano_cohorte"].dropna().unique().tolist())
    neg_ini = float(neg.loc[anos_ord[0]]) if anos_ord else t_global
    neg_fim = float(neg.loc[anos_ord[-1]]) if anos_ord else t_global
    return {
        "n": n,
        "t_defas_neg": t_global,
        "neg_por_ano": neg.to_dict(),
        "inde_med_ano": inde_m.to_dict(),
        "ieg_med_ano": ieg_m.to_dict(),
        "ida_med_ano": ida_m.to_dict(),
        "ips_med_ano": ips_m.to_dict(),
        "neg_primeiro_ano": neg_ini,
        "neg_ultimo_ano": neg_fim,
        "anos": anos_ord,
        "inde_mediano_global": float(full["inde_cohorte"].median()),
        "ieg_mediano_global": float(full["ieg"].median()),
        "ida_mediano_global": float(full["ida"].median()),
        "ips_mediano_global": float(full["ips"].median()),
    }


def gerar_insights(filt: pd.DataFrame, full: pd.DataFrame) -> list[tuple[str, str]]:
    """Gera (markdown, tipo) para callouts Streamlit."""
    if filt.empty:
        return [("Nenhum registro com os filtros atuais. Inclua ao menos um ano e um gênero.", "error")]

    S = _painel_stats(full)
    n_f, n_g = len(filt), len(full)
    share = n_f / n_g
    t_f = float((filt["defasagem"] < 0).mean())
    t_g = S["t_defas_neg"]

    bullets: list[tuple[str, str]] = []

    # Âncora: o que o painel completo mostra (dados reais)
    if share >= 0.97:
        bullets.append(
            (
                f"No painel **{S['n']:,}** registros (RA × ano), **{S['t_defas_neg']:.0%}** dos casos têm **fase abaixo da ideal** "
                f"(defasagem negativa). Entre **{int(S['anos'][0])}** e **{int(S['anos'][-1])}**, essa taxa **cai** de "
                f"**{S['neg_primeiro_ano']:.0%}** para **{S['neg_ultimo_ano']:.0%}**, enquanto o **INDE mediano** sobe de "
                f"**{S['inde_med_ano'].get(S['anos'][0], float('nan')):.2f}** para "
                f"**{S['inde_med_ano'].get(S['anos'][-1], float('nan')):.2f}** — leitura compatível com **melhora agregada** "
                "no recorte disponível (não implica causalidade).",
                "info",
            )
        )
        # IPS 2023 mais baixo que vizinhos (efeito de coorte / instrumento)
        ips_por = S["ips_med_ano"]
        if len(S["anos"]) >= 2 and all(y in ips_por for y in S["anos"]):
            y23 = 2023 if 2023 in ips_por else None
            if y23 is not None:
                v23 = ips_por[y23]
                outros = [ips_por[y] for y in S["anos"] if y != 2023 and y in ips_por]
                if outros and v23 < min(outros) - 0.5:
                    msg_ips = (
                        "**IPS em 2023:** a mediana de IPS no ano **2023** (**{:.1f}**) fica abaixo dos demais anos do painel no mesmo arquivo — "
                        "pode refletir mudança de instrumento, cobertura de avaliação ou composição de turmas; **não** interprete só como piora psíquica."
                    ).format(v23)
                    bullets.append((msg_ips, "info"))

    # Recorte vs média global
    if share < 0.97:
        if t_f > t_g + 0.05:
            bullets.append(
                (
                    f"**Defasagem:** neste recorte, **{t_f:.0%}** com fase abaixo da ideal, acima da média global (**{t_g:.0%}**). "
                    "Priorize diagnóstico de nível e reforço escolar neste grupo.",
                    "warning",
                )
            )
        elif t_f < t_g - 0.05:
            bullets.append(
                (
                    f"**Defasagem:** **{t_f:.0%}** com fase abaixo da ideal, **abaixo** da média global (**{t_g:.0%}**). "
                    "Recorte mais alinhado à fase ideal; útil como referência de boas práticas.",
                    "success",
                )
            )

    # Um único ano selecionado no filtro
    anos_f = sorted(filt["ano_cohorte"].dropna().unique().tolist())
    if len(anos_f) == 1:
        y = int(anos_f[0])
        ref = S["neg_por_ano"].get(y, t_f)
        bullets.append(
            (
                f"**Somente {y}:** **{(filt['defasagem'] < 0).mean():.0%}** com defasagem negativa neste filtro "
                f"(na base completa, o ano **{y}** está em **{ref:.0%}** antes de outros recortes).",
                "info",
            )
        )

    # IEG e IDA vs medianas globais
    if filt["ieg"].notna().any() and filt["ida"].notna().any():
        m_ieg_f = float(filt["ieg"].median())
        m_ida_f = float(filt["ida"].median())
        m_ieg_g = S["ieg_mediano_global"]
        m_ida_g = S["ida_mediano_global"]
        if m_ieg_f < m_ieg_g - 0.35 and m_ida_f < m_ida_g - 0.25:
            bullets.append(
                (
                    f"**Engajamento e desempenho:** medianas **IEG {m_ieg_f:.1f}** (global {m_ieg_g:.1f}) e **IDA {m_ida_f:.1f}** "
                    f"(global {m_ida_g:.1f}) — abaixo do painel típico; revisar lições, frequência e dificuldade das avaliações.",
                    "warning",
                )
            )
        if m_ieg_f > m_ieg_g + 0.35 and m_ida_f > m_ida_g + 0.25:
            bullets.append(
                (
                    f"**Engajamento e desempenho:** **IEG {m_ieg_f:.1f}** e **IDA {m_ida_f:.1f}** acima das medianas globais "
                    f"({m_ieg_g:.1f} / {m_ida_g:.1f}). Bom candidato a mentoria entre pares.",
                    "success",
                )
            )

    # INDE
    if filt["inde_cohorte"].notna().any():
        mi_f = float(filt["inde_cohorte"].median())
        mi_g = S["inde_mediano_global"]
        if mi_f < mi_g - 0.2:
            bullets.append(
                (
                    f"**INDE (variável `inde_cohorte`):** mediana **{mi_f:.2f}** neste filtro vs **{mi_g:.2f}** em todo o painel — "
                    "avaliar peso de IDA, IEG, IPS e IPP conforme a fase (ver documentação PEDE).",
                    "info",
                )
            )
        if mi_f > mi_g + 0.2:
            bullets.append(
                (
                    f"**INDE (`inde_cohorte`):** mediana **{mi_f:.2f}** acima do painel (**{mi_g:.2f}**).",
                    "success",
                )
            )

    # IPS + defasagem conjunta
    if filt["ips"].notna().any():
        ips_med = float(filt["ips"].median())
        ips_glob = S["ips_mediano_global"]
        if ips_med < ips_glob - 0.4 and t_f > 0.5:
            bullets.append(
                (
                    "**Psicossocial:** IPS mediano mais baixo que o painel **e** mais da metade do recorte com defasagem negativa — "
                    "alinhamento entre psicologia e pedagógico recomendado.",
                    "warning",
                )
            )

    bullets.append(
        (
            f"**Recorte atual:** **{n_f:,}** linhas (**{share:.0%}** da base de **{n_g:,}**).",
            "info",
        )
    )

    return bullets[:7]
