# TECH_CHALLENGE_IBOVESPA_FINAL

# 🔍 Tech Challenge — Fase 2  
## Previsão de Tendência do IBOVESPA (Alta ou Baixa)

Este repositório contém a entrega final do **Tech Challenge - Fase 2** do curso de pós-graduação em Data Analytics, no qual o objetivo principal foi desenvolver um modelo preditivo capaz de prever a direção do fechamento do índice **IBOVESPA** no próximo dia: se ele irá **subir (↑)** ou **cair (↓)**.

---

## 🎯 Objetivo do Projeto

Desenvolver um modelo com acurácia mínima de **75%** no conjunto de teste, utilizando dados históricos do IBOVESPA. A previsão deve considerar se o fechamento do índice no dia seguinte será maior ou menor que o do dia atual.

---

## 📊 Fonte dos Dados

Os dados utilizados são históricos diários do IBOVESPA e foram obtidos através da biblioteca Yfinance:  
'''yf.download('^BVSP', period='25y', interval='1d')'''

- Período: diário
- Intervalo utilizado: mais de 25 anos
- Últimos 30 dias reservados como conjunto de teste
  
---

## 🧪 Metodologia

Abaixo está o resumo técnico do projeto:

### 🔹 Aquisição e Exploração dos Dados
- Dados históricos diários do IBOVESPA foram coletados com um intervalo superior a 2 anos.
- Análise exploratória para identificar padrões e sazonalidades.

### 🔹 Engenharia de Atributos
- Criação de features como:
  - Variação percentual (1, 3 e 5 dias)
  - Médias móveis (curtas e médias)
  - Tendência do dia anterior
  - Volume de negociação

### 🔹 Pré-processamento
- Conversão de datas e ordenação crescente
- Cálculo dos retornos percentuais: `ret_1d`, `ret_3d`, `ret_5d`
- Cálculo das médias móveis: `sma_3`, `sma_7`
- Geração da variável `target` binária:
  - `1` se o fechamento do próximo dia for maior
  - `0` caso contrário

### 🔹 Modelos Treinados
- **Regressão Logística**
- **XGBoost** (modelo escolhido)

Todos os modelos foram avaliados com base em:
- Acurácia
- F1-Score
- Matriz de confusão

### 🔹 Resultados
- O modelo **XGBoost** atingiu acurácia **superior a 75%** no conjunto de teste.
- Atributos mais relevantes: `ret_1d`, `sma_3`, `ret_3d`, `ret_5d`.

---

## 🛠️ Tecnologias Utilizadas

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 📎 Entregáveis

| Tipo | Link |
|------|------|
| 🔗 Apresentação Storytelling (PDF/Slides) | [(https://www.canva.com/design/DAGuBU9a834/PeNM00ikv5V-ngJTknioCw/edit?utm_content=DAGuBU9a834&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)] |
| 🎥 Vídeo Explicativo (YouTube ou Drive)   | [(https://drive.google.com/file/d/1PhkiSxvmLScWulDmQ50aPqw9GyZErKd6/view?usp=sharing)] |

## 👩‍🏫 Alunas Responsáveis

**Nomes:**
* Beatriz Calvalcante
* Isabelle Corsi
* Jamile Ribeiro
* Larissa Souza
* Mariana Caldas
  
---
## Curso: Pós-graduação em Data Analytics – FIAP  

**Fase:** 02  








