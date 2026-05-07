"""Textos integrais das perguntas do case (exibição nas abas do Streamlit)."""

QUESTIONS_TEXT: list[str] = [
    """**1. Adequação do nível (IAN)**  
Qual é o perfil geral de defasagem dos alunos (IAN) e como ele evolui ao longo do ano?  
(Ex.: quantos alunos estão moderadamente ou severamente defasados?)""",
    """**2. Desempenho acadêmico (IDA)**  
O desempenho acadêmico médio (IDA) está melhorando, estagnado ou caindo ao longo das fases e anos?""",
    """**3. Engajamento nas atividades (IEG)**  
O grau de engajamento dos alunos (IEG) tem relação direta com seus indicadores de desempenho (IDA) e do ponto de virada (IPV)?""",
    """**4. Autoavaliação (IAA)**  
As percepções dos alunos sobre si mesmos (IAA) são coerentes com seu desempenho real (IDA) e engajamento (IEG)?""",
    """**5. Aspectos psicossociais (IPS)**  
Há padrões psicossociais (IPS) que antecedem quedas de desempenho acadêmico ou de engajamento?""",
    """**6. Aspectos psicopedagógicos (IPP)**  
As avaliações psicopedagógicas (IPP) confirmam ou contradizem a defasagem identificada pelo IAN?""",
    """**7. Ponto de virada (IPV)**  
Quais comportamentos — acadêmicos, emocionais ou de engajamento — mais influenciam o IPV ao longo do tempo?""",
    """**8. Multidimensionalidade dos indicadores**  
Quais combinações de indicadores (IDA + IEG + IPS + IPP) elevam mais a nota global do aluno (INDE)?""",
    """**9. Previsão de risco com Machine Learning**  
Quais padrões nos indicadores permitem identificar alunos em risco antes de queda no desempenho ou aumento da defasagem? Construa um modelo preditivo que mostre uma probabilidade do aluno ou aluna entrar em risco de defasagem.""",
    """**10. Efetividade do programa**  
Os indicadores mostram melhora consistente ao longo do ciclo nas diferentes fases (Quartzo, Ágata, Ametista e Topázio), confirmando o impacto real do programa?""",
]
