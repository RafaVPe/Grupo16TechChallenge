#%% md
# 
#%%
# Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# Estilo dos gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
rcParams['figure.figsize'] = 12, 6
rcParams['font.size'] = 10
#%%
#Arquivo PNAD limpo
caminho_junho = r"C:\Users\gabri\Documents\projeto_pnad\PNAD_Junho.csv"

with open(caminho_junho, "r", encoding="latin1") as f:
    for i in range(10):
        print(f.readline())
#%%
caminho_junho = r"C:\Users\gabri\Documents\projeto_pnad\PNAD_Junho.csv"
caminho_julho = r"C:\Users\gabri\Documents\projeto_pnad\PNAD_Julho.csv"

df_junho = pd.read_csv(caminho_junho, sep=",", encoding="latin1")
df_julho = pd.read_csv(caminho_julho, sep=",", encoding="latin1")

# Adicionar coluna do mês
df_junho["mes"] = "Junho"
df_julho["mes"] = "Julho"

# Unir os dois meses
df = pd.concat([df_junho, df_julho], ignore_index=True)

# Verificar colunas disponíveis
print("Colunas disponíveis:", df.columns.tolist())
df.head()
#%%
#Dicionário PNAD
descricao_variaveis = {
    "B0011": "Febre",
    "B0012": "Tosse",
    "B0013": "Dor de garganta",
    "B0014": "Dificuldade para respirar",
    "B0015": "Dor de cabeça",
    "B0016": "Dor no peito",
    "B0017": "Náusea ou vômito",
    "B0018": "Diarreia",
    "B0019": "Perda de olfato ou paladar",
    "B00111": "Outro sintoma",
    "A005": "Escolaridade",
    "C001": "Trabalho na semana",
    "C013": "Home office",
    "C011A12": "Renda do trabalho",
    "D0053": "Auxílio emergencial",
    "F002A1": "Mora com criança",
    "F002A2": "Mora com idoso",
    "F002A3": "Mora com pessoa com doença crônica"
}

# Lista de variáveis
variaveis_sintomas = list(descricao_variaveis.keys())[:10]
variaveis_todas = list(descricao_variaveis.keys())
#%%
print("Colunas disponíveis:", df.columns.tolist())
#%%
df_f = df[df["Variavel"].isin(variaveis_todas)].copy()

print("Dataset filtrado com sucesso!")
print("Formato final:", df_f.shape)
df_f.head()
#%%
print(df.shape)
df.head()
#%%
#Criar df_f com as variáveis relevantes
variaveis_todas = list(descricao_variaveis.keys())

df_f = df[df["Variavel"].isin(variaveis_todas)].copy()

print("Dataset filtrado com sucesso!")
print("Formato final:", df_f.shape)
#%%
# Dicionário Escolaridade A005
categorias_escolaridade = {
    1.0: "Sem instrução",
    2.0: "Fundamental incompleto",
    3.0: "Fundamental completo",
    4.0: "Médio incompleto",
    5.0: "Médio completo",
    6.0: "Superior incompleto",
    7.0: "Superior completo",
    8.0: "Pós-graduação"
}

#Frequência Escolaridade A005

codigo_variavel = "A005"
df_var = df_f[df_f["Variavel"] == codigo_variavel].copy()

# contar os valores da coluna 'Valor'
freq_valores = df_var["Valor"].value_counts().sort_index()

# substituir os códigos pelo dicionário
freq_valores.index = [categorias_escolaridade.get(valor, valor) for valor in freq_valores.index]

# nome amigável
nome_variavel = descricao_variaveis.get(codigo_variavel, codigo_variavel)

# gráfico
plt.figure(figsize=(10, 5))
sns.barplot(x=freq_valores.index, y=freq_valores.values, palette="rocket")
plt.title(f"Distribuição dos valores — {nome_variavel}", fontsize=14, fontweight="bold")
plt.xlabel("Categoria de Escolaridade", fontsize=12)
plt.ylabel("Quantidade de Registros", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#%%
# Escolaridade Junho × Julho

codigo_variavel = "A005"
df_var = df_f[df_f["Variavel"] == codigo_variavel].copy()

# Dicionário da variável A005
categorias_escolaridade = {
    1.0: "Sem instrução",
    2.0: "Fundamental incompleto",
    3.0: "Fundamental completo",
    4.0: "Médio incompleto",
    5.0: "Médio completo",
    6.0: "Superior incompleto",
    7.0: "Superior completo",
    8.0: "Pós-graduação"
}


df_var["Valor"] = df_var["Valor"].map(categorias_escolaridade)

# Contar valores por mês
freq_por_mes = df_var.groupby(["mes", "Valor"]).size().reset_index(name="Quantidade")

# comparando Junho × Julho
plt.figure(figsize=(12, 6))
sns.barplot(data=freq_por_mes, x="Valor", y="Quantidade", hue="mes", palette="Set2")
plt.title("Distribuição da Escolaridade por Mês — PNAD COVID", fontsize=14, fontweight="bold")
plt.xlabel("Categoria de Escolaridade", fontsize=12)
plt.ylabel("Quantidade de Registros", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title="Mês")
plt.tight_layout()
plt.show()
#%%
#Proporção de Escolaridade

codigo_variavel = "A005"
df_var = df_f[df_f["Variavel"] == codigo_variavel].copy()

# Dicionário de categorias
categorias_escolaridade = {
    1.0: "Sem instrução",
    2.0: "Fundamental incompleto",
    3.0: "Fundamental completo",
    4.0: "Médio incompleto",
    5.0: "Médio completo",
    6.0: "Superior incompleto",
    7.0: "Superior completo",
    8.0: "Pós-graduação"
}


df_var["Valor"] = df_var["Valor"].map(categorias_escolaridade)

# Calcular proporção
freq_valores = df_var["Valor"].value_counts(normalize=True).sort_index() * 100


plt.figure(figsize=(8, 8))
plt.pie(freq_valores.values, labels=freq_valores.index, autopct="%.1f%%", startangle=90, colors=sns.color_palette("pastel"))
plt.title("Proporção de Escolaridade — PNAD COVID", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
#%% md
# uma grande falta na educação que pode estar associada com a não informação da doença
#%%
# C001 — Situação de Trabalho
df_trabalho = df[df["Variavel"] == "C001"].copy()
freq_trabalho = df_trabalho["Valor"].value_counts().sort_index()

# C013 — Modalidade de Trabalho
df_homeoffice = df[df["Variavel"] == "C013"].copy()
freq_homeoffice = df_homeoffice["Valor"].value_counts().sort_index()


fig, axs = plt.subplots(1, 2, figsize=(14, 6))

#Situação de Trabalho
sns.barplot(x=freq_trabalho.index.astype(str), y=freq_trabalho.values, ax=axs[0], palette="Blues")
axs[0].set_title("Situação de Trabalho (valores reais)")
axs[0].set_ylabel("Quantidade")
axs[0].set_xlabel("Código da base")
for i, valor in enumerate(freq_trabalho.values):
    axs[0].text(i, valor + 5, str(valor), ha='center', va='bottom', fontsize=10)

# Modalidade de Trabalho
sns.barplot(x=freq_homeoffice.index.astype(str), y=freq_homeoffice.values, ax=axs[1], palette="Greens")
axs[1].set_title("Modalidade de Trabalho (valores reais)")
axs[1].set_ylabel("Quantidade")
axs[1].set_xlabel("Código da base")
for i, valor in enumerate(freq_homeoffice.values):
    axs[1].text(i, valor + 5, str(valor), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()