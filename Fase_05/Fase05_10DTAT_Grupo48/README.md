# Datathon — Passos Mágicos (PEDE)

## Ambiente

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dados e limpeza

- CSVs na raiz: `BASE DE DADOS PEDE 2024 - DATATHON - PEDE2022.csv` (e 2023, 2024).
- Unificação: `src/pede_cleaning.py` → `data_processed/pede_unificado.parquet`.
- Notebook: `notebooks/01_limpeza_unificada.ipynb`.

## Modelo de risco

- **Automático:** ao subir o Streamlit, se faltar `data_processed/pede_unificado.parquet` ou `App/models/risk_defasagem.joblib`, o app **gera** (primeira execução pode levar ~1 minuto).
- **Manual (opcional):** `python scripts/train_model.py` ou `python scripts/train_model.py --force` para forçar retreino.
- Notebook: `notebooks/02_modelagem_risco.ipynb`.

## App Streamlit

Na raiz do repositório:

```powershell
python -m streamlit run App/DATATHON_Inicio.py
```

- **Início:** `DATATHON_Inicio.py` — apresentação do trabalho, equipe e botões para as demais telas.
- **PEDE — 10 perguntas:** `pages/1_PEDE_10_perguntas.py` — dez abas com gráficos e textos.
- Menu **Pages**: **Insights** e **Modelo / Predição** (carrega o `.joblib`).

## Especificação

Ver `SPEC.md` para decisões de escopo e deploy (Community Cloud — pendente).

### Streamlit Community Cloud

O app tenta **gravar** parquet e joblib no repositório na primeira execução. Se o ambiente for **somente leitura**,
faça commit de `data_processed/pede_unificado.parquet` e `App/models/risk_defasagem.joblib` **ou** use secrets/cache
conforme a política da disciplina.
