# Previsão de churn e análise de risco

Projeto de portfólio que compara modelos de classificação e disponibiliza uma interface Streamlit para consultar risco de cancelamento, investigar previsões e exportar resultados.

O objetivo de negócio é apoiar a priorização de clientes para uma ação de retenção. O projeto não mede o efeito de uma oferta nem demonstra redução de churn em uma operação real.

## Aplicação e análise

- [Link da demonstração Streamlit](https://churn-ai-agent-gust.streamlit.app/) — a disponibilidade depende da hospedagem.
- [Notebook de treinamento e avaliação](notebooks/churn_prevision_ia_agent.ipynb).
- [CSV de exemplo para a aplicação](data/raw/sample_input.csv).
- [Template de entrada](data/raw/template_upload.csv).

Apesar do nome do repositório, a implementação atual usa modelos scikit-learn, SHAP e regras de recomendação. Não há integração com LLM ou agente autônomo no fluxo da aplicação.

## Dados e problema

O arquivo [data/raw/churn.csv](data/raw/churn.csv) contém a base de telecomunicações utilizada na análise. As saídas salvas no notebook registram 7.043 clientes, sendo 1.869 da classe de cancelamento e 5.174 da classe de permanência. O alvo é `Churn`; `customerID` é removido das variáveis preditoras.

A modelagem utiliza 19 atributos de perfil, serviços contratados, tempo de relacionamento e cobrança. A procedência externa e a licença da versão do CSV precisam ser documentadas antes de sua redistribuição em outros projetos.

## Método e avaliação

1. Inspeção de tipos, nulos e duplicidades; conversão de `TotalCharges` para numérico, com preenchimento de ausências por zero antes da separação.
2. Divisão aleatória estratificada de 80% para treino e 20% para teste, com `random_state=42`: 5.634 e 1.409 registros nas saídas salvas.
3. Pipelines com imputação e padronização de atributos numéricos e imputação/one-hot encoding de categóricos. Na versão corrigida do pré-processador, `SeniorCitizen` é tratado como categórico.
4. Comparação entre regressão logística e Random Forest, ambas com `class_weight="balanced"`.
5. Avaliação de acurácia, ROC AUC e relatório por classe; análise adicional com limiar de 0,30 e explicação da Random Forest com SHAP.

### Resultados registrados no notebook

Os valores abaixo são transcritos das saídas versionadas, sem nova execução nesta revisão documental. Precisão e recall referem-se à classe de cancelamento, no limiar de 0,50.

| Modelo | Acurácia | ROC AUC | Precisão | Recall |
| --- | ---: | ---: | ---: | ---: |
| Regressão logística | 0,7410 | 0,8416 | 0,51 | 0,79 |
| Random Forest | 0,7658 | 0,8423 | 0,54 | 0,75 |

No limiar de 0,30, o recall registrado sobe para 0,93 na regressão logística e 0,89 na Random Forest; a precisão cai para 0,43 e 0,45, respectivamente. Isso demonstra o compromisso entre alcançar mais clientes que cancelam e contatar clientes que permaneceriam.

O notebook compara modelos e limiares sobre o mesmo teste; falta um conjunto de validação independente para selecionar essas decisões. Os resultados devem ser tratados como exploratórios. Ainda é necessário registrar uma comparação formal com um baseline simples.

## O que a aplicação implementa

- Upload de CSV ou XLSX, validação das colunas esperadas e conversão de tipos.
- Uso de `models/churn_pipeline_rf.pkl` para gerar escores de churn.
- Distribuição dos clientes por faixa de risco, ranking e download de resultados.
- Importância de atributos e SHAP opcional, calculado em até 200 registros do arquivo enviado.
- Explicações individuais dentro da amostra usada pelo SHAP e recomendações baseadas em regras.

As faixas do aplicativo são: baixo risco para escores menores que 0,40; médio de 0,40 até menos de 0,70; alto a partir de 0,70. São regras da interface, distintas do limiar de classificação de 0,30 explorado no notebook. Os escores ainda precisam de uma avaliação de calibração.

## Executar localmente

O repositório indica Python 3.12 em `.python-version` e `runtime.txt`.

```bash
git clone https://github.com/Gustavocorreard/churn-ai-agent.git
cd churn-ai-agent
python -m venv .venv
```

Ative o ambiente:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS ou Linux
source .venv/bin/activate
```

Instale as dependências e os complementos importados pela interface:

```bash
python -m pip install -r requirements.txt
python -m pip install matplotlib openpyxl
python -m streamlit run streamlit_app.py
```

Execute os comandos a partir da raiz do repositório para que os caminhos relativos funcionem. Comece enviando `data/raw/sample_input.csv`, que contém as 19 colunas obrigatórias. `Churn` não é necessário para inferência.

Para explorar o treinamento:

```bash
python -m pip install jupyterlab
python -m jupyter lab notebooks/churn_prevision_ia_agent.ipynb
```

Se o kernel iniciar em `notebooks/`, ajuste a célula de leitura para `DATA_PATH = "../data/raw/churn.csv"`. Ajuste também os caminhos de exportação se quiser gravar na pasta `models/` da raiz.

### Limitações de reprodução conhecidas

- `requirements.txt` fixa scikit-learn 1.4.2, enquanto `pyproject.toml` declara 1.4.0. É necessário unificar o ambiente e confirmar a versão usada nos artefatos `.pkl`; a instalação e a compatibilidade dos modelos não foram revalidadas nesta revisão.
- O notebook anuncia regressão logística como modelo principal em uma seção, mas exporta a Random Forest e a aplicação carrega a Random Forest. O comportamento da interface está descrito acima.
- O notebook trata `SeniorCitizen` como categoria, enquanto o validador da interface o converte para numérico. Essa diferença deve ser alinhada e testada antes de considerar o fluxo reproduzível.
- Ausências em `TotalCharges` são preenchidas com zero no notebook e com a mediana do lote enviado na interface. É necessário unificar essa regra com o pré-processamento do treinamento.
- A análise temporal, a seleção de limiar em validação independente e o monitoramento de desempenho ainda fazem parte das próximas etapas.

## Estrutura principal

```text
data/raw/                                  Base, exemplo e template
models/churn_pipeline_rf.pkl                Pipeline carregado pela aplicação
models/churn_pipeline_lr.pkl                Pipeline alternativo exportado
notebooks/churn_prevision_ia_agent.ipynb     Treinamento e avaliação
src/validation.py                          Validação da entrada na aplicação
streamlit_app.py                           Interface, inferência e SHAP
requirements.txt                          Dependências
```

## Próximas etapas

- Unificar ambiente, contrato de entrada e pré-processamento entre treino e inferência.
- Separar validação de teste e comparar baseline, métricas de ranking e calibração.
- Escolher o limiar com capacidade de contato e custos explícitos.
- Planejar um experimento para medir efeito real da campanha de retenção.
- Adicionar testes, rastreabilidade e monitoramento antes de uso operacional.

SHAP descreve contribuições para as previsões do modelo; não demonstra causas de cancelamento ou o efeito de uma intervenção comercial.

## Autor

[Gustavo Correard no LinkedIn](https://www.linkedin.com/in/gustavo-correard/) · [GitHub](https://github.com/Gustavocorreard)
