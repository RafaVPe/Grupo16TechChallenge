# Tech Challenger Fase 04 

### Desafio 
 Imagina que você foi contratado como cientista de dados de 
um hospital e tem o desafio de desenvolver um modelo de Machine Learning 
para auxiliar os médicos e médicas a prever se uma pessoa pode ter 
obesidade.  


A obesidade é uma condição médica caracterizada pelo acúmulo 
excessivo de gordura corporal, a ponto de prejudicar a saúde. Esse problema 
tem se tornado cada vez mais prevalente em todo o mundo, atingindo pessoas 
de todas as idades e classes sociais. As causas da obesidade são multifatoriais 
e envolvem uma combinação de fatores genéticos, ambientais e 
comportamentais.   


Utilizando a base de dados disponibilizada neste desafio em 
“obesity.csv”, desenvolva um modelo preditivo e crie um sistema preditivo para 
auxiliar a tomada de decisão da equipe médica a diagnosticar a obesidade. 

### Objetivo

Como entregável, você precisar desenvolver toda a pipeline de Machine 
Learning, garantindo a entrega de todos os seguintes requisitos listados a seguir 
que serão avaliados na sua entrega: 


- Pipeline de machine learning demonstrando toda a etapa de feature 
engineering e treinamento do modelo. 
- Modelo com assertividade acima de 75%. 
- Realizar o deploy do seu modelo em uma aplicação preditiva 
utilizando o Streamlit. 
- Construir uma visão analítica em um painel com principais insights 
obtidos com o estudo sobre obesidade para trazer insights para a 
equipe médica. 
- Compartilhar o link da sua aplicação deployada no app do Streamlit 
+ link do painel analítico + link do repositório do seu github com todo 
código desenvolvido em um arquivo .doc ou .txt para realizar o 
upload na plataforma online. 
- Gravar um vídeo mostrando toda a estratégia utilizada e apresentação do 
sistema preditivo (algo em torno de 4min - 10min). Não se esqueça que 
tanto a visão do sistema preditivo quanto o dashboard analítico deve 
ser apresentado em uma visão de negócio.

### Perguntas norteadoras

- Qual o problema de negócio o hospital quer resolver? 
O hospital quer por meio de um modelo de machine learning conseguir ajudar os médicos a identificar quais pessoas estão propensas a ter obsidade, isso facilitaria atendimentos, tratamentos, precauções e auxilios a longo prazo para cada individuo.

- Qual decisão médica será apoiada pelo modelo?
Identificar individuos com obesidade. 

- Tipo de problema? 
Como a variavel target tem N respostas, seria uma classificação multiclasses, mesmo se construirmos uma coluna de IMC para facilitar a identificação do grau de obesidade de cada individuo. 

- Qual métrica de sucesso? 
Assertividade acima de 75%.

- Qual será o entregável final para o médico? 
Um modelo online feito com streamlit para que os médicos possam colocar as informações de cada individuo e o modelo prever o grau/nível de obesidade de cada indivíduo. Também será entregue uma visão análitica construida com principais insights obtidos com o estudo.

