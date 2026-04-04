# 🚀 Churn Prediction AI Agent

Este projeto tem como objetivo prever o risco de churn (cancelamento de clientes) utilizando Machine Learning e disponibilizar insights acionáveis através de uma interface interativa em Streamlit.

A solução foi construída com foco em aplicação real de negócio, permitindo que empresas identifiquem clientes com maior probabilidade de cancelamento e tomem decisões estratégicas para retenção.

---


## 🌐 Acesse a Aplicação

Você pode testar o projeto em produção através do link abaixo:

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-red?logo=streamlit)](https://churn-ai-agent-gust.streamlit.app/)


---


## 📌 Problema de Negócio

A retenção de clientes é um dos principais desafios das empresas.

Adquirir novos clientes pode custar até 5x mais do que manter os atuais. Portanto, prever quais clientes estão propensos a churn permite:

* Reduzir perdas de receita
* Direcionar ações de retenção
* Otimizar campanhas de marketing
* Aumentar o LTV (Lifetime Value)

---

## 🎯 Objetivo do Projeto

Construir uma solução completa que:

* Preveja a probabilidade de churn por cliente
* Explique os fatores que influenciam a decisão do modelo
* Gere insights acionáveis
* Permita upload de novas bases de dados
* Disponibilize resultados para download

---

## 🧠 Abordagem

O projeto foi dividido nas seguintes etapas:

### 1. 🔍 Exploração e Tratamento dos Dados

* Limpeza e padronização
* Conversão de variáveis categóricas
* Tratamento de valores ausentes
* Feature engineering

### 2. ⚙️ Modelagem

* Pipeline com pré-processamento + modelo
* Algoritmos testados:

  * Logistic Regression
  * Random Forest
* Avaliação com:

  * Accuracy
  * Precision / Recall
  * ROC AUC

### 3. 📊 Interpretação do Modelo

* Uso de técnicas explicáveis (ex: SHAP ou análise de features)
* Identificação das variáveis mais relevantes

Exemplo de fatores de churn identificados:

* InternetService (Fiber)
* PaymentMethod (Electronic Check)
* PaperlessBilling

---

## 🖥️ Aplicação (Streamlit)

A aplicação permite que o usuário:

* Faça upload de uma base de clientes
* Visualize:

  * Ranking de risco de churn
  * Probabilidade por cliente
  * Distribuições e gráficos
* Gere insights automáticos
* Exporte os resultados em CSV

---

## 📂 Estrutura do Projeto

churn-ai-agent/

├── data/                # Dados utilizados
├── models/              # Modelo treinado (pipeline.pkl)
├── notebooks/           # Análises exploratórias
├── src/                 # Scripts auxiliares
│   ├── preprocessing.py
│   ├── validation.py
│   └── utils.py

├── streamlit_app.py     # Aplicação principal
├── requirements.txt
└── README.md

---

## ⚙️ Tecnologias Utilizadas

* Python
* Pandas / NumPy
* Scikit-learn
* SHAP (ou interpretação de features)
* Streamlit
* Joblib

---

## 📈 Resultados

O modelo apresentou:

* Accuracy: ~80%
* ROC AUC: ~0.84

Com boa capacidade de identificar clientes com alto risco de churn.

---

## 💡 Insights de Negócio

Alguns padrões identificados:

* Clientes com Fiber optic possuem maior risco de churn
* Pagamentos via Electronic Check indicam maior cancelamento
* Clientes com Paperless Billing tendem a churnar mais

Ações recomendadas:

* Criar campanhas específicas para esses perfis
* Oferecer incentivos de retenção
* Revisar experiência do cliente nesses segmentos

---

## 🚀 Como Rodar o Projeto

### 1. Clone o repositório

git clone https://github.com/seu-usuario/churn-ai-agent.git
cd churn-ai-agent

### 2. Instale as dependências

pip install -r requirements.txt

### 3. Execute o app

streamlit run streamlit_app.py

---

## 📤 Próximos Passos (Melhorias)

* Deploy em cloud (AWS / GCP)
* Integração com APIs
* Monitoramento do modelo (ML Ops)
* Atualização automática com novos dados
* Explicações mais avançadas com SHAP interativo

---

## 📢 Diferencial do Projeto

Este projeto não é apenas um modelo de Machine Learning.

Ele entrega uma solução completa com:

* Aplicação utilizável por empresas
* Foco em impacto de negócio
* Explicabilidade do modelo
* Interface amigável

---

## 👨‍💻 Autor

Gustavo Correard

LinkedIn: https://www.linkedin.com/in/gustavo-correard/
GitHub: https://github.com/Gustavocorreard

---

## ⭐ Se esse projeto te ajudou, deixe uma estrela!
