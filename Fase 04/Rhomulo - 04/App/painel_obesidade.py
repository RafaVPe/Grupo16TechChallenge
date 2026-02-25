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
import joblib

# Caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Obesity.csv"
PIPELINE_PATHS = [
    BASE_DIR / "App" / "pipeline_obesidade_tech_challenge_4.joblib",
    BASE_DIR / "pipeline_obesidade_tech_challenge_4.joblib",
]

# Configuração da página
st.set_page_config(
    page_title="Sistema de Apoio ao Diagnóstico de Obesidade",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def carregar_pipeline(_path_key):
    """Carrega o pipeline de predição do Tech Challenge 4. _path_key invalida cache ao atualizar o arquivo."""
    for p in PIPELINE_PATHS:
        if p.exists():
            return joblib.load(p)
    return None

def _get_pipeline_path_key():
    """Retorna caminho e mtime do pipeline para invalidar cache quando o arquivo mudar."""
    for p in PIPELINE_PATHS:
        if p.exists():
            return f"{p}_{p.stat().st_mtime}"
    return "none"

def prever_obesidade(paciente_dict, pipeline, model=None):
    """Aplica o mesmo pré-processamento do treino e prediz. model=None usa pipeline['model']."""
    scale_limits = {"FCVC": (1, 3), "NCP": (1, 4), "CH2O": (1, 3), "FAF": (0, 3), "TUE": (0, 2)}
    paciente = pd.DataFrame([paciente_dict])
    if "BMI" in pipeline["num_cols"]:
        paciente["BMI"] = paciente["Weight"] / (paciente["Height"] ** 2)
    for col, (low, high) in scale_limits.items():
        if col in paciente.columns:
            paciente[col] = np.clip(np.round(float(paciente[col].iloc[0])), low, high).astype(int)
    ordinal_map = pipeline.get("ordinal_map", {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3})
    paciente["CAEC"] = paciente["CAEC"].map(ordinal_map).fillna(0).astype(int)
    paciente["CALC"] = paciente["CALC"].map(ordinal_map).fillna(0).astype(int)
    num_cols = pipeline["num_cols"]
    cat_cols = pipeline["cat_cols"]
    feature_names = pipeline["feature_names"]
    scaler = pipeline["scaler"]
    m = model if model is not None else pipeline["model"]
    le = pipeline["label_encoder"]
    X_num_p = paciente[num_cols]
    X_cat_p = pd.get_dummies(paciente[cat_cols], drop_first=True, dtype=int)
    X_p = pd.concat([X_num_p.reset_index(drop=True), X_cat_p.reset_index(drop=True)], axis=1)
    X_p = X_p.reindex(columns=feature_names, fill_value=0)
    X_p[num_cols] = scaler.transform(X_p[num_cols])
    pred = m.predict(X_p)[0]
    return le.inverse_transform([pred])[0]

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

# Navegação
st.sidebar.header("Sistema de Apoio ao Diagnóstico de Obesidade")
pagina = st.sidebar.radio(
    "Navegação",
    ["Painel analítico", "Predição pelo modelo"],
    label_visibility="collapsed"
)

# Mapeamentos PT -> EN (apenas visualização, backend em inglês)
SIM_NAO = {"Sim": "yes", "Não": "no"}
CAEC_CALC = {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
MTRANS_PT = {"Carro": "Automobile", "Moto": "Motorbike", "Bicicleta": "Bike", "Transporte público": "Public_Transportation", "A pé": "Walking"}
OBESIDADE_PT = {"Insufficient_Weight": "Abaixo do peso", "Normal_Weight": "Peso normal", "Overweight_Level_I": "Sobrepeso Nível I",
    "Overweight_Level_II": "Sobrepeso Nível II", "Obesity_Type_I": "Obesidade Tipo I", "Obesity_Type_II": "Obesidade Tipo II",
    "Obesity_Type_III": "Obesidade Tipo III"}


def gerar_insights_dinamicos(df_filtrado, df_total, faixa_etaria, genero_filtro):
    """Gera insights em texto com base nas métricas calculadas do conjunto filtrado."""
    if df_filtrado.empty:
        return ["Nenhum registro corresponde aos filtros selecionados."]

    n = len(df_filtrado)
    min_idade, max_idade = faixa_etaria
    genero_text = ""
    if genero_filtro == "Female":
        genero_text = ", mulheres"
    elif genero_filtro == "Male":
        genero_text = ", homens"

    # Métricas do filtrado
    pct_ob = (df_filtrado["Obesity"].str.contains("Obesity", na=False).sum() / n * 100) if n else 0
    pct_sob = (df_filtrado["Obesity"].str.contains("Overweight|Obesity", na=False).sum() / n * 100) if n else 0
    bmi_medio = df_filtrado["BMI"].mean()
    faf_medio = df_filtrado["FAF"].mean()
    fcvc_medio = df_filtrado["FCVC"].mean()
    pct_hist = (df_filtrado["family_history"] == "yes").sum() / n * 100 if n else 0
    pct_favc = (df_filtrado["FAVC"] == "yes").sum() / n * 100 if n else 0
    pct_scc = (df_filtrado["SCC"] == "yes").sum() / n * 100 if n else 0

    # Métricas totais para comparação
    n_total = len(df_total)
    pct_ob_total = (df_total["Obesity"].str.contains("Obesity", na=False).sum() / n_total * 100) if n_total else 0
    pct_sob_total = (df_total["Obesity"].str.contains("Overweight|Obesity", na=False).sum() / n_total * 100) if n_total else 0
    bmi_total = df_total["BMI"].mean()
    faf_total = df_total["FAF"].mean()
    fcvc_total = df_total["FCVC"].mean()

    insights = []

    # 1. Visão geral do perfil
    insights.append(
        f"**Perfil selecionado (idade {min_idade}–{max_idade} anos{genero_text}):** "
        f"{n:,} registros. **{pct_ob:.1f}%** apresentam obesidade (tipos I/II/III) e **{pct_sob:.1f}%** estão em sobrepeso ou obesidade."
    )

    # 2. Comparação com a amostra total
    if pct_ob > pct_ob_total + 2:
        insights.append(f"Esta faixa apresenta **proporção maior** de obesidade ({pct_ob:.1f}% vs {pct_ob_total:.1f}% na amostra total). Atenção redobrada em triagens.")
    elif pct_ob < pct_ob_total - 2:
        insights.append(f"Esta faixa apresenta **proporção menor** de obesidade ({pct_ob:.1f}% vs {pct_ob_total:.1f}% na amostra total).")

    # 3. IMC médio
    insights.append(f"O **IMC médio** nesta faixa é **{bmi_medio:.1f}** (vs {bmi_total:.1f} na amostra total).")

    # 4. Atividade física
    if faf_medio < faf_total - 0.15:
        insights.append(f"**Atividade física (FAF):** A média é **{faf_medio:.2f}** — **abaixo** da amostra total ({faf_total:.2f}). Reforçar incentivo à atividade física.")
    elif faf_medio > faf_total + 0.15:
        insights.append(f"**Atividade física (FAF):** A média é **{faf_medio:.2f}** — acima da amostra total ({faf_total:.2f}).")
    else:
        insights.append(f"**Atividade física (FAF):** Média de **{faf_medio:.2f}** (escala 0–3), similar à amostra total.")

    # 5. Consumo de vegetais
    insights.append(f"**Consumo de vegetais (FCVC):** Média de **{fcvc_medio:.2f}** (escala 1–3). {'Maior que a média geral' if fcvc_medio > fcvc_total else 'Similar ou abaixo da média geral'}.")

    # 6. Fatores de risco
    if pct_hist > 50:
        insights.append(f"**Histórico familiar:** **{pct_hist:.1f}%** têm histórico de sobrepeso/obesidade — reforçar acompanhamento e orientação.")
    if pct_favc > 50:
        insights.append(f"**Consumo calórico (FAVC):** **{pct_favc:.1f}%** consomem frequentemente alimentos muito calóricos — ponto de intervenção dietética.")
    if pct_scc > 20:
        insights.append(f"**Monitoramento calórico:** {pct_scc:.1f}% monitoram calorias — pode ser ferramenta útil quando orientada pela equipe.")

    # 7. Comparação por gênero (quando "Todos")
    if genero_filtro == "Todos" and "Female" in df_filtrado["Gender"].values and "Male" in df_filtrado["Gender"].values:
        m_mask = df_filtrado["Gender"] == "Male"
        f_mask = df_filtrado["Gender"] == "Female"
        n_m, n_f = m_mask.sum(), f_mask.sum()
        pct_ob_m = (df_filtrado.loc[m_mask, "Obesity"].str.contains("Obesity", na=False).sum() / n_m * 100) if n_m else 0
        pct_ob_f = (df_filtrado.loc[f_mask, "Obesity"].str.contains("Obesity", na=False).sum() / n_f * 100) if n_f else 0
        insights.append(f"**Por gênero nesta faixa:** obesidade em **{pct_ob_m:.1f}%** dos homens e **{pct_ob_f:.1f}%** das mulheres.")

    return insights


def gerar_dicas_predicao(predicao, paciente):
    """Gera dicas personalizadas com base na classificação e no perfil do paciente."""
    dicas = []
    faf = int(paciente.get("FAF", 1))
    mtrans = paciente.get("MTRANS", "")
    fcvc = int(paciente.get("FCVC", 2))
    favc = paciente.get("FAVC", "no")
    tue = int(paciente.get("TUE", 1))
    ch2o = int(paciente.get("CH2O", 2))
    family_history = paciente.get("family_history", "no")

    # Dicas por classificação
    if "Insufficient_Weight" in predicao:
        dicas.append("**Manutenção saudável:** Buscar ganho de peso com alimentação balanceada e atividade física moderada. Acompanhamento nutricional pode ajudar.")
    elif "Normal_Weight" in predicao:
        dicas.append("**Manutenção:** Manter hábitos saudáveis — alimentação equilibrada, atividade física regular e hidratação.")
    elif "Overweight" in predicao or "Obesity" in predicao:
        dicas.append("**Atenção ao peso:** Recomenda-se acompanhamento médico ou nutricional. Pequenas mudanças no dia a dia podem fazer diferença.")

    # Transporte — incentivo a bicicleta/a pé (quando usa carro, ônibus ou moto)
    if mtrans in ("Automobile", "Public_Transportation", "Motorbike") and ("Overweight" in predicao or "Obesity" in predicao or faf <= 1):
        dicas.append("**Transporte:** Considerar **usar bicicleta ou ir a pé** em trajetos curtos quando possível — aumenta a atividade física no dia a dia sem precisar de academia.")

    # Atividade física baixa
    if faf <= 1 and ("Overweight" in predicao or "Obesity" in predicao):
        dicas.append("**Atividade física:** Aumentar a frequência — começar com **caminhadas de 20–30 min** ou 2–3× por semana já traz benefícios.")
    elif faf == 0:
        dicas.append("**Atividade física:** Incluir atividade física na rotina — mesmo caminhadas leves ajudam na saúde geral.")

    # Consumo de vegetais baixo
    if fcvc <= 1 and ("Overweight" in predicao or "Obesity" in predicao):
        dicas.append("**Vegetais:** Incluir **mais vegetais** nas refeições — saladas, legumes no almoço e jantar. Começar por uma porção extra já ajuda.")

    # Alimentos calóricos
    if favc == "yes" and ("Overweight" in predicao or "Obesity" in predicao):
        dicas.append("**Alimentação:** Reduzir a frequência de alimentos muito calóricos — trocar por opções mais nutritivas e menos processadas.")

    # Tempo em eletrônicos
    if tue >= 2 and ("Overweight" in predicao or "Obesity" in predicao):
        dicas.append("**Dispositivos eletrônicos:** Reduzir o tempo em telas quando possível — está associado a menos movimento. Pausas para alongar ou caminhar ajudam.")

    # Hidratação
    if ch2o == 1:
        dicas.append("**Hidratação:** Aumentar o consumo de água — cerca de **1,5 a 2 litros por dia** auxilia no metabolismo e na saciedade.")

    # Histórico familiar
    if family_history == "yes" and ("Overweight" in predicao or "Obesity" in predicao):
        dicas.append("**Histórico familiar:** Com histórico de sobrepeso na família, o acompanhamento regular com médico ou nutricionista é especialmente recomendado.")

    # Obesidade tipos II/III — reforço
    if "Obesity_Type_II" in predicao or "Obesity_Type_III" in predicao:
        dicas.append("**Importante:** Em níveis mais elevados de obesidade, o acompanhamento médico é fundamental para um plano adequado e seguro.")

    return dicas if dicas else ["Manter hábitos saudáveis e consultar um profissional de saúde para orientações personalizadas."]


if pagina == "Predição pelo modelo":
    st.title("🩺 Predição de Obesidade")
    st.markdown("Preencha os dados do paciente para obter a predição do modelo.")
    pipeline = carregar_pipeline(_get_pipeline_path_key())
    if pipeline is None:
        st.error("Pipeline não encontrado. Execute o notebook `pipeline_obesidade_ml_TC4.ipynb` e salve o arquivo `pipeline_obesidade_tech_challenge_4.joblib` na pasta App ou na raiz do projeto.")
    else:
        with st.form("form_predicao"):
            st.subheader("Dados do paciente")
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox("Gênero", ["Female", "Male"], format_func=lambda x: "Feminino" if x == "Female" else "Masculino")
                age = st.number_input("Idade (anos)", min_value=14, max_value=61, value=25)
                height = st.number_input("Altura (m)", min_value=1.0, max_value=2.2, value=1.70, step=0.01, format="%.2f")
                weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1, format="%.1f")
            with c2:
                family_history = SIM_NAO[st.selectbox("Histórico familiar de sobrepeso", ["Sim", "Não"])]
                favc = SIM_NAO[st.selectbox("Consumo frequente de alimentos calóricos", ["Sim", "Não"])]
                fcvc = st.select_slider("Frequência de vegetais (1=raro, 3=sempre)", options=[1, 2, 3], value=2)
                ncp = st.select_slider("Nº refeições principais por dia", options=[1, 2, 3, 4], value=3)
                caec = CAEC_CALC[st.selectbox("Consumo entre refeições", ["Não", "Às vezes", "Frequentemente", "Sempre"])]
            with c3:
                smoke = SIM_NAO[st.selectbox("Fuma", ["Sim", "Não"])]
                ch2o = st.select_slider("Consumo de água por dia (1–3)", options=[1, 2, 3], value=2)
                scc = SIM_NAO[st.selectbox("Monitora calorias", ["Sim", "Não"])]
                faf = st.select_slider("Atividade física por semana (0–3)", options=[0, 1, 2, 3], value=1)
                tue = st.select_slider("Tempo em eletrônicos por dia (0–2)", options=[0, 1, 2], value=1)
                calc = CAEC_CALC[st.selectbox("Consumo de álcool", ["Não", "Às vezes", "Frequentemente", "Sempre"])]
                mtrans = MTRANS_PT[st.selectbox("Meio de transporte", ["Carro", "Moto", "Bicicleta", "Transporte público", "A pé"])]
            submitted = st.form_submit_button("Obter predição")
        if submitted:
            paciente = {
                "Gender": gender, "Age": age, "Height": height, "Weight": weight,
                "family_history": family_history, "FAVC": favc, "FCVC": fcvc, "NCP": ncp,
                "CAEC": caec, "SMOKE": smoke, "CH2O": ch2o, "SCC": scc, "FAF": faf, "TUE": tue,
                "CALC": calc, "MTRANS": mtrans
            }
            # Predição dos dois modelos (se disponíveis)
            if "model_lr" in pipeline and "model_rf" in pipeline:
                pred_lr = prever_obesidade(paciente, pipeline, pipeline["model_lr"])
                pred_rf = prever_obesidade(paciente, pipeline, pipeline["model_rf"])
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.success(f"**Regressão Logística:** {OBESIDADE_PT.get(pred_lr, pred_lr)}")
                with col_res2:
                    st.success(f"**Random Forest:** {OBESIDADE_PT.get(pred_rf, pred_rf)}")
                if pred_lr == pred_rf:
                    st.info("Ambos os modelos concordam na classificação.")
                else:
                    st.warning("Os modelos apresentam classificações diferentes. Considere avaliar com critério clínico.")
                pred_para_dicas = pred_rf  # usa RF como referência para dicas
            else:
                pred = prever_obesidade(paciente, pipeline)
                st.success(f"**Classificação predita:** {OBESIDADE_PT.get(pred, pred)}")
                pred_para_dicas = pred

            # Dicas personalizadas
            dicas = gerar_dicas_predicao(pred_para_dicas, paciente)
            with st.expander("💡 Dicas personalizadas", expanded=True):
                for d in dicas:
                    st.markdown(f"- {d}")

            st.caption("O resultado é uma ferramenta de apoio à decisão. O diagnóstico final deve ser feito pelo médico.")
else:
    # --- Painel analítico ---
    st.title("📊 Painel Analítico – Estudo de Obesidade")
    st.markdown("**Principais insights obtidos com o estudo para apoio à equipe médica.**")
    st.divider()

    # Sidebar: filtros opcionais
    st.sidebar.header("Filtros")
    faixa_etaria = st.sidebar.slider("Faixa etária (anos)", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
    genero_opts = ["Todos", "Female", "Male"]
    genero_filtro = st.sidebar.selectbox("Gênero", genero_opts, format_func=lambda x: {"Todos": "Todos", "Female": "Feminino", "Male": "Masculino"}[x])

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
        st.subheader("Insights dinâmicos para a equipe médica")
        st.caption("Atualizados conforme os filtros de faixa etária e gênero selecionados na barra lateral.")
        insights = gerar_insights_dinamicos(df_filtrado, df, faixa_etaria, genero_filtro)
        for i, texto in enumerate(insights):
            st.markdown(f"- {texto}")
        st.divider()
        st.markdown("""
        **Orientações gerais (independentes dos filtros):**
        - O modelo de Machine Learning (Random Forest / Regressão Logística) pode ser usado como **ferramenta de apoio** à triagem, nunca como substituto do diagnóstico médico. Validar com exame clínico e critérios da OMS.
        """)
        st.info("Os filtros na barra lateral (idade e gênero) aplicam-se a todas as visualizações e aos insights acima.")

st.sidebar.divider()
st.sidebar.caption("Painel Obesidade – Tech Challenge Fase 04")
