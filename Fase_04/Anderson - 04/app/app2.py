import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Sistema Clínico - Avaliação de Obesidade",
    page_icon="🏥",
    layout="centered"
)

st.markdown("""
<style>
body {
    background-color: #f4f9fb;
}

h1 {
    color: #0a3d62;
    text-align: center;
}

.stButton>button {
    background-color: #0a3d62;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

.stProgress > div > div > div > div {
    background-color: #0a3d62;
}
</style>
""", unsafe_allow_html=True)

st.title("🏥 Sistema de Avaliação sobre Risco de Obesidade")
st.write("Preencha as informações abaixo para realizar a avaliação clínica automatizada.")

st.divider()


# Leitura do modelo de ML
model = joblib.load("models/modelo_obesidade.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


#Formulário para frevisão
with st.form("prediction_form"):

    st.subheader("👤 Informações Gerais")

    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Idade", min_value=1, max_value=100, value=25)
        Height = st.number_input("Altura (metros)", min_value=1.0, max_value=2.50, value=1.70)
        Weight = st.number_input("Peso (kg)", min_value=20.0, max_value=250.0, value=70.0)

    with col2:
        gender_options = {"Masculino": "Male", "Feminino": "Female"}
        Gender = gender_options[
            st.selectbox("Gênero", list(gender_options.keys()))
        ]

        yes_no = {"Sim": "yes", "Não": "no"}

        family_history = yes_no[
            st.selectbox("Histórico familiar de obesidade?", list(yes_no.keys()))
        ]

    st.divider()
    st.subheader("🥗 Hábitos Alimentares")

    col3, col4 = st.columns(2)

    with col3:
        FCVC = {
            "Raramente": 1,
            "Às vezes": 2,
            "Frequentemente": 3
        }[st.selectbox("Frequência de consumo de vegetais", 
                       ["Raramente", "Às vezes", "Frequentemente"])]

        NCP = {
            "1 refeição": 1,
            "2 refeições": 2,
            "3 refeições": 3,
            "4 ou mais": 4
        }[st.selectbox("Número de refeições principais por dia", 
                       ["1 refeição", "2 refeições", "3 refeições", "4 ou mais"])]

        CAEC = {
            "Não": "no",
            "Às vezes": "Sometimes",
            "Frequentemente": "Frequently",
            "Sempre": "Always"
        }[st.selectbox("Consome lanches entre refeições?", 
                       ["Não", "Às vezes", "Frequentemente", "Sempre"])]
        
        yes_no = {"Sim": "yes", "Não": "no"}

        FAVC = yes_no[
            st.selectbox("Consome alimentos altamente calóricos com frequência?", list(yes_no.keys()))
        ]

    with col4:
        CH2O = {
            "Menos de 1L": 1,
            "1-2L": 2,
            "Mais de 2L": 3
        }[st.selectbox("Consumo diário de água", 
                       ["Menos de 1L", "1-2L", "Mais de 2L"])]

        CALC = {
            "Não": "no",
            "Às vezes": "Sometimes",
            "Frequentemente": "Frequently",
            "Sempre": "Always"
        }[st.selectbox("Consumo de bebida alcoólica", 
                       ["Não", "Às vezes", "Frequentemente", "Sempre"])]

    st.divider()
    st.subheader("🏃 Estilo de Vida")

    col5, col6 = st.columns(2)

    with col5:
        FAF = {
            "Nenhuma": 0,
            "1-2x por semana": 1,
            "3-4x por semana": 2,
            "5x ou mais": 3
        }[st.selectbox("Frequência semanal de atividade física", 
                       ["Nenhuma", "1-2x por semana", "3-4x por semana", "5x ou mais"])]

        TUE = {
            "0-2h por dia": 0,
            "3-5h por dia": 1,
            "Mais de 5h por dia": 2
        }[st.selectbox("Tempo diário em dispositivos eletrônicos", 
                       ["0-2h por dia", "3-5h por dia", "Mais de 5h por dia"])]

    with col6:
        SMOKE = yes_no[
            st.selectbox("Fuma?", list(yes_no.keys()))
        ]

        SCC = yes_no[
            st.selectbox("Monitora ingestão calórica?", list(yes_no.keys()))
        ]

        MTRANS = {
            "Carro": "Automobile",
            "Moto": "Motorbike",
            "Bicicleta": "Bike",
            "Transporte público": "Public_Transportation",
            "Caminhando": "Walking"
        }[st.selectbox("Meio de transporte habitual", 
                       ["Carro", "Moto", "Bicicleta", "Transporte público", "Caminhando"])]

    submit = st.form_submit_button("🔎 Realizar Avaliação")


# Predição do modelo

if submit:

    input_data = pd.DataFrame({
        "Age": [Age],
        "Height": [Height],
        "Weight": [Weight],
        "FCVC": [FCVC],
        "NCP": [NCP],
        "CH2O": [CH2O],
        "FAF": [FAF],
        "TUE": [TUE],
        "Gender": [Gender],
        "family_history": [family_history],
        "FAVC": [FAVC],
        "CAEC": [CAEC],
        "SMOKE": [SMOKE],
        "SCC": [SCC],
        "CALC": [CALC],
        "MTRANS": [MTRANS]
    })

    prediction = model.predict(input_data)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    proba = model.predict_proba(input_data)
    confidence = max(proba[0]) * 100


# Resultados 
    st.divider()
    st.subheader("📊 Resultado da Avaliação")

    low_risk = ["Insufficient_Weight", "Normal_Weight"]
    medium_risk = ["Overweight_Level_I", "Overweight_Level_II"]
    high_risk = ["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]

    if predicted_label in low_risk:
        st.success(f"Classificação: {predicted_label}")
        st.success("Nível de risco: Baixo")
        st.info("Recomendação: Manter um estilo de vida saudável e monitorar a rotina.")
    elif predicted_label in medium_risk:
        st.warning(f"Classificação: {predicted_label}")
        st.warning("Nível de risco: Moderado")
        st.warning("Recomendação: Considerar aconselhamento nutricional e um plano de atividade física.")

    else:
        st.error(f"Classificação: {predicted_label}")
        st.error("Nível de risco: Alto")
        st.error("Recomendação: Avaliação clínica e intervenção estruturada recomendada.")

    st.write(f"Confiança do modelo: {confidence:.2f}%")
    st.progress(int(confidence))

    st.info("⚠️ Este sistema é um apoio à decisão clínica e não substitui avaliação médica especializada.")