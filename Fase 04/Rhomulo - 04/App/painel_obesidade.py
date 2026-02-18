"""
Painel analítico - Estudo de Obesidade
Insights para a equipe médica.
Execute na raiz do projeto: streamlit run App/painel_obesidade.py
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Caminho do CSV: pasta data na raiz do projeto (pai da pasta App)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Obesity.csv"

# Configuração da página
st.set_page_config(
    page_title="Painel Obesidade | Insights para Equipe Médica",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def carregar_dados():
    df = pd.read_csv(DATA_PATH)
    # Sem ID por pessoa, linhas iguais podem ser pessoas diferentes com mesmo perfil — não removemos duplicatas
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    # Arredondar escalas conforme dicionário
    for col, (low, high) in [("FCVC", (1, 3)), ("NCP", (1, 4)), ("CH2O", (1, 3)), ("FAF", (0, 3)), ("TUE", (0, 2))]:
        if col in df.columns:
            df[col] = np.clip(np.round(df[col].astype(float)), low, high)
    return df

df = carregar_dados()

# Títulos e introdução
st.title("📊 Painel Analítico – Estudo de Obesidade")
st.markdown("**Principais insights obtidos com o estudo para apoio à equipe médica.**")
st.divider()

# Sidebar: filtros opcionais
st.sidebar.header("Filtros")
faixa_etaria = st.sidebar.slider("Faixa etária (anos)", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
genero_filtro = st.sidebar.selectbox("Gênero", ["Todos", "Female", "Male"])

mask = (df["Age"] >= faixa_etaria[0]) & (df["Age"] <= faixa_etaria[1])
if genero_filtro != "Todos":
    mask = mask & (df["Gender"] == genero_filtro)
df_filtrado = df[mask].copy()

# --- KPIs no topo ---
st.subheader("Visão geral")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total de registros", f"{len(df_filtrado):,}")
with col2:
    st.metric("Idade média", f"{df_filtrado['Age'].mean():.1f} anos")
with col3:
    st.metric("IMC médio", f"{df_filtrado['BMI'].mean():.1f}")
with col4:
    pct_obesidade = (df_filtrado["Obesity"].str.contains("Obesity", na=False).sum() / len(df_filtrado) * 100) if len(df_filtrado) else 0
    st.metric("% com obesidade (I/II/III)", f"{pct_obesidade:.1f}%")
with col5:
    pct_sobrepeso_ob = ((df_filtrado["Obesity"].str.contains("Overweight|Obesity", na=False).sum()) / len(df_filtrado) * 100) if len(df_filtrado) else 0
    st.metric("% sobrepeso + obesidade", f"{pct_sobrepeso_ob:.1f}%")

st.divider()

# --- Abas: Distribuições, Comportamento, Insights ---
tab1, tab2, tab3, tab4 = st.tabs(["Distribuição da obesidade", "Perfil e IMC", "Fatores comportamentais", "Insights para a equipe"])

with tab1:
    st.subheader("Distribuição das classes de peso (Obesity)")
    contagem = df_filtrado["Obesity"].value_counts().sort_index()
    fig = px.bar(
        x=contagem.index, y=contagem.values,
        labels={"x": "Classificação", "y": "Quantidade"},
        color=contagem.values, color_continuous_scale="Blues",
        text=contagem.values
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45, coloraxis_showscale=False)
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Proporção por classe (%)")
        fig_pie = px.pie(values=contagem.values, names=contagem.index, hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_b:
        st.subheader("Obesidade por gênero")
        cross = pd.crosstab(df_filtrado["Gender"], df_filtrado["Obesity"], normalize="index") * 100
        fig_gen = px.imshow(cross, x=cross.columns, y=cross.index, labels=dict(x="Classe", y="Gênero", color="%"), color_continuous_scale="Blues", aspect="auto")
        fig_gen.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_gen, use_container_width=True)

with tab2:
    st.subheader("IMC por classe de obesidade")
    fig_bmi = px.box(df_filtrado, x="Obesity", y="BMI", color="Obesity", points="outliers")
    fig_bmi.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig_bmi, use_container_width=True)

    st.subheader("Idade vs IMC (cor por classe)")
    fig_scatter = px.scatter(df_filtrado, x="Age", y="BMI", color="Obesity", opacity=0.6, hover_data=["Weight", "Height"])
    st.plotly_chart(fig_scatter, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Peso médio por classe (kg)")
        peso_medio = df_filtrado.groupby("Obesity")["Weight"].mean().sort_index()
        fig_peso = px.bar(x=peso_medio.index, y=peso_medio.values, labels={"x": "Classe", "y": "Peso médio (kg)"})
        fig_peso.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_peso, use_container_width=True)
    with col_d:
        st.subheader("Idade média por classe")
        idade_media = df_filtrado.groupby("Obesity")["Age"].mean().sort_index()
        fig_idade = px.bar(x=idade_media.index, y=idade_media.values, labels={"x": "Classe", "y": "Idade média"})
        fig_idade.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_idade, use_container_width=True)

with tab3:
    st.subheader("Fatores associados ao peso")
    comport = ["family_history", "FAVC", "FAF", "SCC", "SMOKE"]
    for var in comport:
        if var not in df_filtrado.columns:
            continue
        cross = pd.crosstab(df_filtrado["Obesity"], df_filtrado[var], normalize="index") * 100
        fig_c = px.bar(
            cross.T, barmode="group", title=f"Classe de peso × {var}",
            labels={"value": "% na classe", "Obesity": "Classe"}
        )
        fig_c.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_c, use_container_width=True)

    st.subheader("Atividade física (FAF) e consumo de vegetais (FCVC) por classe")
    agg = df_filtrado.groupby("Obesity").agg(FAF_medio=("FAF", "mean"), FCVC_medio=("FCVC", "mean")).reset_index()
    fig_agg = make_subplots(specs=[[{"secondary_y": True}]])
    fig_agg.add_trace(go.Bar(x=agg["Obesity"], y=agg["FAF_medio"], name="FAF (atividade física)", marker_color="steelblue"), secondary_y=False)
    fig_agg.add_trace(go.Scatter(x=agg["Obesity"], y=agg["FCVC_medio"], name="FCVC (vegetais)", marker_color="darkgreen"), secondary_y=True)
    fig_agg.update_layout(xaxis_tickangle=-45, title="Médias por classe")
    fig_agg.update_yaxes(title_text="FAF (0–3)", secondary_y=False)
    fig_agg.update_yaxes(title_text="FCVC (1–3)", secondary_y=True)
    st.plotly_chart(fig_agg, use_container_width=True)

with tab4:
    st.subheader("Principais insights para a equipe médica")
    st.markdown("""
    - **IMC é o principal discriminador:** As classes de obesidade do dataset seguem faixas de IMC (abaixo do peso, normal, sobrepeso I/II, obesidade I/II/III). O modelo preditivo e as análises confirmam que peso e altura (e IMC derivado) são os fatores que mais explicam a classificação.

    - **Histórico familiar:** Pacientes com histórico familiar de sobrepeso/obesidade tendem a apresentar maior proporção nas categorias de sobrepeso e obesidade. Vale reforçar orientação e acompanhamento nesses casos.

    - **Consumo de alimentos muito calóricos (FAVC):** Maior frequência de consumo de alimentos muito calóricos está associada a maiores proporções nas classes de sobrepeso e obesidade. Ponto de intervenção dietética.

    - **Atividade física (FAF):** Classes de peso mais altas tendem a ter menor média de frequência de atividade física semanal. Incentivo à atividade física é relevante em todos os níveis, principalmente em sobrepeso e obesidade.

    - **Consumo de vegetais (FCVC):** Maior frequência de consumo de vegetais nas refeições tende a estar associada a classes de peso mais saudáveis. Reforçar orientação de dieta balanceada.

    - **Monitoramento calórico (SCC):** O hábito de monitorar calorias está mais presente em alguns perfis; pode ser uma ferramenta útil quando orientada pela equipe.

    - **Uso do modelo preditivo:** O modelo de Machine Learning (Random Forest / Regressão Logística / Gradient Boosting) treinado neste estudo pode ser usado como **ferramenta de apoio** à triagem ou à decisão clínica, nunca como substituto do diagnóstico médico. Recomenda-se validar com exame clínico e critérios da OMS.
    """)
    st.info("Os filtros na barra lateral (idade e gênero) aplicam-se a todas as visualizações deste painel.")

st.sidebar.divider()
st.sidebar.caption("Painel Obesidade – Tech Challenge Fase 04")
