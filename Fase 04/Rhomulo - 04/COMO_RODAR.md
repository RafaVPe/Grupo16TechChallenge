# Como rodar tudo que eu fiz ai

## Onde rodar

Sempre no **PowerShell** (ou CMD acho que vcs tudo usa windows), na **pasta raiz do projeto**:

```
...\Grupo16TechChallenge\Fase 04\Rhomulo - 04
```

Ou seja, o mesmo lugar onde estão as pastas `App` e `data` e o arquivo `requirements.txt`.

---

## 1. Instalar dependências (só na primeira vez)

Se `pip` não for reconhecido, use `python -m pip`:

```powershell
python -m pip install streamlit pandas numpy plotly
```

Ou, se tiver o arquivo de requisitos:

```powershell
python -m pip install -r requirements.txt
```

---

## 2. Subir o painel do streamlit

Na **mesma pasta** (raiz do projeto), execute:

```powershell
python -m streamlit run App/painel_obesidade.py
```

**Importante:** o script está em `App/painel_obesidade.py`, por isso o comando usa `App/painel_obesidade.py`.

O navegador deve abrir em `http://localhost:8501`. Para parar o ele, use `Ctrl+C` no terminal.

---

## Resumo

| O quê              | Onde / Comando |
|--------------------|----------------|
| Abrir terminal     | Na pasta **Rhomulo - 04** (raiz do projeto) |
| Instalar pacotes   | `python -m pip install -r requirements.txt` |
| Rodar o painel     | `python -m streamlit run App/painel_obesidade.py` |