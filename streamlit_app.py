import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from src.validation import validate_input_data, REQUIRED_COLUMNS

# =========================
# Configuração da página
# =========================
st.set_page_config(
    page_title="Churn Prediction & Insights",
    layout="wide",
    page_icon="📊"
)

# =========================
# caminhos e mapeamentos
# =========================
TEMPLATE_PATH = "data/raw/template_upload.csv"
SAMPLE_PATH   = "data/raw/sample_input.csv"
PIPELINE_PATH = "models/churn_pipeline_rf.pkl"

COLUMN_LABELS_PT = {
    "Unnamed: 0": "Índice",
    "customerID": "ID do Cliente",
    "gender": "Gênero",
    "SeniorCitizen": "Idoso",
    "Partner": "Possui Parceiro(a)",
    "Dependents": "Possui Dependentes",
    "tenure": "Tempo de Contrato",
    "PhoneService": "Serviço de Telefone",
    "MultipleLines": "Múltiplas Linhas",
    "InternetService": "Serviço de Internet",
    "OnlineSecurity": "Segurança Online",
    "OnlineBackup": "Backup Online",
    "DeviceProtection": "Proteção do Dispositivo",
    "TechSupport": "Suporte Técnico",
    "StreamingTV": "Streaming de TV",
    "StreamingMovies": "Streaming de Filmes",
    "Contract": "Tipo de Contrato",
    "PaperlessBilling": "Fatura Digital",
    "PaymentMethod": "Método de Pagamento",
    "MonthlyCharges": "Cobrança Mensal",
    "TotalCharges": "Gasto Total",
    "churn_probability": "Probabilidade de Churn",
    "churn_probability_pct": "Probabilidade de Churn (%)",
    "risk_level": "Nível de Risco",
}

RISK_ORDER = ["Alto risco", "Médio risco", "Baixo risco"]


# =========================
# Funções auxiliares
# =========================
def classify_risk(prob: float) -> str:
    if prob >= 0.7:
        return "Alto risco"
    if prob >= 0.4:
        return "Médio risco"
    return "Baixo risco"


def translate_for_display(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_LABELS_PT)


def translate_feature_name(name: str) -> str:
    name = str(name).replace("num__", "").replace("cat__", "")

    mapping = {
        "gender": "Gênero",
        "SeniorCitizen": "Idoso",
        "Partner": "Possui Parceiro(a)",
        "Dependents": "Possui Dependentes",
        "tenure": "Tempo de Contrato",
        "PhoneService": "Serviço de Telefone",
        "MultipleLines": "Múltiplas Linhas",
        "InternetService": "Serviço de Internet",
        "OnlineSecurity": "Segurança Online",
        "OnlineBackup": "Backup Online",
        "DeviceProtection": "Proteção do Dispositivo",
        "TechSupport": "Suporte Técnico",
        "StreamingTV": "Streaming de TV",
        "StreamingMovies": "Streaming de Filmes",
        "Contract": "Tipo de Contrato",
        "PaperlessBilling": "Fatura Digital",
        "PaymentMethod": "Método de Pagamento",
        "MonthlyCharges": "Cobrança Mensal",
        "TotalCharges": "Gasto Total",
    }

    for key, value in mapping.items():
        if key in name:
            name = name.replace(key, value)

    return name


def generate_general_recommendation(high_pct: float, medium_pct: float, low_pct: float) -> str:
    if high_pct >= 0.20:
        return """
A base apresenta uma concentração relevante de clientes em **alto risco de churn**.

**Próximos passos recomendados:**
- priorizar ações imediatas de retenção nesse grupo
- realizar contato proativo com ofertas, benefícios ou condições especiais
- investigar os perfis mais críticos e os fatores associados ao risco
- acompanhar esse grupo com maior frequência
"""
    if medium_pct >= 0.30:
        return """
Existe uma concentração importante de clientes em **médio risco**.

**Próximos passos recomendados:**
- atuar de forma preventiva antes que esses clientes avancem para alto risco
- segmentar esse grupo por perfil de contrato, cobrança e tempo de casa
- testar campanhas de relacionamento e retenção
- monitorar esse grupo nas próximas análises
"""
    return """
A maior parte da base está concentrada em **baixo risco**, indicando um cenário relativamente saudável.

**Próximos passos recomendados:**
- manter ações preventivas de relacionamento
- monitorar continuamente os grupos de médio e alto risco
- acompanhar mudanças no comportamento da base ao longo do tempo
- criar uma rotina recorrente de análise de churn
"""


def build_top_insights(results: pd.DataFrame) -> list[str]:
    insights = []
    high_risk_df = results[results["risk_level"] == "Alto risco"].copy()

    if high_risk_df.empty:
        return ["Não foram encontrados clientes classificados como alto risco nesta base."]

    if "Contract" in high_risk_df.columns and high_risk_df["Contract"].notna().any():
        top_contract = high_risk_df["Contract"].mode().iloc[0]
        contract_pct = (high_risk_df["Contract"] == top_contract).mean()
        insights.append(
            f"Entre os clientes de alto risco, o tipo de contrato mais frequente é **{top_contract}** "
            f"({contract_pct:.1%} desse grupo)."
        )

    if "PaymentMethod" in high_risk_df.columns and high_risk_df["PaymentMethod"].notna().any():
        top_payment = high_risk_df["PaymentMethod"].mode().iloc[0]
        payment_pct = (high_risk_df["PaymentMethod"] == top_payment).mean()
        insights.append(
            f"O método de pagamento mais comum entre os clientes de alto risco é **{top_payment}** "
            f"({payment_pct:.1%} desse grupo)."
        )

    if "InternetService" in high_risk_df.columns and high_risk_df["InternetService"].notna().any():
        top_internet = high_risk_df["InternetService"].mode().iloc[0]
        internet_pct = (high_risk_df["InternetService"] == top_internet).mean()
        insights.append(
            f"O serviço de internet mais presente entre os clientes de alto risco é **{top_internet}** "
            f"({internet_pct:.1%} desse grupo)."
        )

    if "tenure" in high_risk_df.columns and high_risk_df["tenure"].notna().any():
        avg_tenure = high_risk_df["tenure"].mean()
        insights.append(
            f"Os clientes de alto risco têm, em média, **{avg_tenure:.1f} meses** de permanência."
        )

    if "MonthlyCharges" in high_risk_df.columns and high_risk_df["MonthlyCharges"].notna().any():
        avg_monthly = high_risk_df["MonthlyCharges"].mean()
        insights.append(
            f"A cobrança mensal média dos clientes de alto risco é de **{avg_monthly:.2f}**."
        )

    return insights[:5]


def get_display_columns(df: pd.DataFrame) -> list[str]:
    preferred_columns = [
        "customerID",
        "gender",
        "tenure",
        "Contract",
        "PaymentMethod",
        "InternetService",
        "MonthlyCharges",
        "TotalCharges",
        "churn_probability_pct",
        "risk_level",
    ]
    return [col for col in preferred_columns if col in df.columns]


def get_feature_names(preprocessor) -> list[str]:
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue

        if hasattr(transformer, "get_feature_names_out"):
            names = transformer.get_feature_names_out(columns)
            feature_names.extend(names)
        else:
            feature_names.extend(columns)

    return feature_names


def build_feature_importance_df(model, preprocessor) -> pd.DataFrame:
    feature_names = get_feature_names(preprocessor)
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    importance_df["feature_pt"] = importance_df["feature"].apply(translate_feature_name)
    return importance_df


@st.cache_data(show_spinner=False)
def build_shap_outputs_cached(_model, _preprocessor, df_model: pd.DataFrame, sample_size: int = 200):
    if len(df_model) > sample_size:
        df_shap = df_model.sample(sample_size, random_state=42).copy()
    else:
        df_shap = df_model.copy()

    X_transformed = _preprocessor.transform(df_shap)
    feature_names = get_feature_names(_preprocessor)

    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values_positive = shap_values[1]
    else:
        if len(shap_values.shape) == 3:
            shap_values_positive = shap_values[:, :, 1]
        else:
            shap_values_positive = shap_values

    mean_abs_shap = abs(shap_values_positive).mean(axis=0)

    shap_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs_shap
    }).sort_values("importance", ascending=False)

    shap_importance_df["feature_pt"] = shap_importance_df["feature"].apply(translate_feature_name)

    return shap_values_positive, shap_importance_df, df_shap, feature_names


def build_individual_shap_df(shap_values_positive, feature_names, selected_index: int) -> pd.DataFrame:
    client_shap_values = shap_values_positive[selected_index]

    client_features = pd.DataFrame({
        "feature": feature_names,
        "shap_value": client_shap_values
    })

    client_features["feature_pt"] = client_features["feature"].apply(translate_feature_name)
    client_features["abs_shap"] = client_features["shap_value"].abs()

    return client_features


def generate_individual_recommendation(top_positive_df: pd.DataFrame) -> str:
    factors = top_positive_df[top_positive_df["shap_value"] > 0]["feature_pt"].tolist()

    if not factors:
        return "Este cliente não apresenta fatores críticos relevantes na explicação individual."

    main_factors = ", ".join(factors[:3])

    return (
        f"Os principais fatores que estão elevando o risco deste cliente são: **{main_factors}**. "
        f"Recomenda-se priorizar ações de retenção e analisar esse perfil com maior atenção."
    )


# =========================
# Cabeçalho
# =========================
st.title("📊 Churn Prediction & Insights")
st.write(
    "Faça upload de uma base de clientes para prever risco de churn, "
    "identificar perfis críticos e apoiar decisões de retenção."
)

st.info(
    """
Este modelo foi treinado com um schema específico.

**Para evitar erros:**
1. Baixe o template oficial
2. Preencha sua base com as mesmas colunas
3. Envie o arquivo para análise
"""
)

# =========================
# Downloads
# =========================
col_download_1, col_download_2 = st.columns(2)

with col_download_1:
    try:
        with open(TEMPLATE_PATH, "rb") as f:
            st.download_button(
                label="📥 Baixar template CSV",
                data=f,
                file_name="template_upload.csv",
                mime="text/csv",
                width="stretch",
            )
    except FileNotFoundError:
        st.warning("Arquivo de template não encontrado.")

with col_download_2:
    try:
        with open(SAMPLE_PATH, "rb") as f:
            st.download_button(
                label="📥 Baixar exemplo preenchido",
                data=f,
                file_name="sample_input.csv",
                mime="text/csv",
                width="stretch",
            )
    except FileNotFoundError:
        st.warning("Arquivo de exemplo não encontrado.")

# =========================
# Carregamento do pipeline
# =========================
try:
    pipeline = joblib.load(PIPELINE_PATH)
except Exception as e:
    st.error(f"Erro ao carregar o pipeline: {e}")
    st.stop()

model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocessor"]

# =========================
# Upload
# =========================
uploaded_file = st.file_uploader("Envie sua base de clientes", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    with st.expander("📄 Ver prévia da base enviada", expanded=False):
        st.dataframe(translate_for_display(df.head(10)), width="stretch")

    validation_result = validate_input_data(df)

    if not validation_result["valid"]:
        st.error("A base enviada não é compatível com o modelo atual.")

        if validation_result["missing_columns"]:
            st.write("### Colunas obrigatórias ausentes")
            st.write(validation_result["missing_columns"])

        if validation_result["type_errors"]:
            st.write("### Problemas encontrados nos tipos das colunas")
            st.write(validation_result["type_errors"])

        st.info("Baixe o template oficial e ajuste sua base antes de tentar novamente.")
        st.stop()

    if validation_result["extra_columns"]:
        st.warning(
            f"As seguintes colunas extras serão ignoradas no modelo: {validation_result['extra_columns']}"
        )

    if validation_result.get("warnings"):
        for warning in validation_result["warnings"]:
            st.warning(warning)

    df_valid = validation_result["normalized_df"].copy()

    with st.expander("🛠️ Ver prévia da base após tratamento", expanded=False):
        st.dataframe(translate_for_display(df_valid.head(10)), width="stretch")

    try:
        df_model = df_valid[REQUIRED_COLUMNS]
    except KeyError as e:
        st.error(f"Erro ao selecionar colunas para o modelo: {e}")
        st.stop()

    try:
        preds = pipeline.predict_proba(df_model)[:, 1]
    except Exception as e:
        st.error(f"Erro ao executar a previsão: {e}")
        st.stop()

    # =========================
    # Resultados
    # =========================
    results = df_valid.copy()
    results["churn_probability"] = preds
    results["churn_probability_pct"] = (results["churn_probability"] * 100).round(2)
    results["risk_level"] = results["churn_probability"].apply(classify_risk)
    results = results.sort_values("churn_probability", ascending=False).reset_index(drop=True)

    total = len(results)
    high_risk = (results["risk_level"] == "Alto risco").sum()
    medium_risk = (results["risk_level"] == "Médio risco").sum()
    low_risk = (results["risk_level"] == "Baixo risco").sum()

    high_pct = high_risk / total if total > 0 else 0
    medium_pct = medium_risk / total if total > 0 else 0
    low_pct = low_risk / total if total > 0 else 0

    # =========================
    # Visão geral
    # =========================
    st.subheader("📊 Visão Geral da Base")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de clientes", f"{total}")
    c2.metric("Alto risco", f"{high_risk} ({high_pct:.1%})")
    c3.metric("Médio risco", f"{medium_risk} ({medium_pct:.1%})")
    c4.metric("Baixo risco", f"{low_risk} ({low_pct:.1%})")

    st.subheader("📈 Distribuição de risco")
    risk_counts = results["risk_level"].value_counts().reindex(RISK_ORDER).fillna(0)
    st.bar_chart(risk_counts)

    st.subheader("📌 Recomendação Geral")
    st.markdown(generate_general_recommendation(high_pct, medium_pct, low_pct))

    # =========================
    # Insights automáticos
    # =========================
    st.subheader("🧠 Insights Automáticos")
    insights = build_top_insights(results)
    for insight in insights:
        st.markdown(f"- {insight}")

    # =========================
    # Feature importance
    # =========================
    st.subheader("📊 Principais fatores que influenciam o churn")
    importance_df = build_feature_importance_df(model, preprocessor)
    top_features = importance_df.head(10)

    st.bar_chart(top_features.set_index("feature_pt")["importance"])

    top_names = top_features["feature_pt"].tolist()
    if len(top_names) >= 3:
        st.markdown(f"""
Os principais fatores que influenciam o churn neste modelo são:

- **{top_names[0]}**
- **{top_names[1]}**
- **{top_names[2]}**

Isso significa que essas variáveis têm maior impacto na probabilidade de um cliente cancelar.
""")

    # =========================
    # SHAP global e individual
    # =========================
    st.subheader("🧩 Explicação global com SHAP")

    
    enable_shap = st.checkbox(
        "Ativar análise SHAP (mais detalhada, porém mais pesada)",
        value=False
    )

    shap_values_positive = None
    feature_names = None

    if enable_shap:
        try:
            with st.spinner("Calculando explicações SHAP..."):
                shap_values_positive, shap_importance_df, df_shap, feature_names = build_shap_outputs_cached(
                    model,
                    preprocessor,
                    df_model,
                    sample_size=200
                )

            st.caption(f"SHAP calculado em {len(df_shap)} registro(s) para melhorar performance.")

            top_shap = shap_importance_df.head(10).copy()
            st.bar_chart(top_shap.set_index("feature_pt")["importance"])

            top_shap_names = top_shap["feature_pt"].tolist()
            if len(top_shap_names) >= 3:
                st.markdown(f"""
As variáveis que mais influenciam as previsões do modelo nesta amostra são:

- **{top_shap_names[0]}**
- **{top_shap_names[1]}**
- **{top_shap_names[2]}**

Essa análise ajuda a entender quais fatores estão tendo maior peso na propensão de churn.
""")
        except Exception as e:
            st.warning(f"Não foi possível gerar a explicação global com SHAP: {e}")
            shap_values_positive = None
            feature_names = None

    if shap_values_positive is not None and feature_names is not None:
        st.subheader("🔍 Explicação individual de um cliente da amostra SHAP")

        max_index = len(shap_values_positive) - 1

        selected_index = st.number_input(
            "Escolha o índice do cliente na amostra SHAP",
            min_value=0,
            max_value=max_index,
            value=0,
            step=1
        )

        client_features = build_individual_shap_df(
            shap_values_positive,
            feature_names,
            int(selected_index)
        )

        top_positive = client_features.sort_values("shap_value", ascending=False).head(5)
        top_negative = client_features.sort_values("shap_value", ascending=True).head(5)

        col_pos, col_neg = st.columns(2)

        with col_pos:
            st.markdown("### Fatores que aumentam o risco")
            shown = False
            for _, row in top_positive.iterrows():
                if row["shap_value"] > 0:
                    st.markdown(f"- **{row['feature_pt']}** (impacto: {row['shap_value']:.4f})")
                    shown = True
            if not shown:
                st.write("Nenhum fator positivo relevante encontrado.")

        with col_neg:
            st.markdown("### Fatores que reduzem o risco")
            shown = False
            for _, row in top_negative.iterrows():
                if row["shap_value"] < 0:
                    st.markdown(f"- **{row['feature_pt']}** (impacto: {row['shap_value']:.4f})")
                    shown = True
            if not shown:
                st.write("Nenhum fator negativo relevante encontrado.")

        st.markdown("### Recomendação individual")
        st.info(generate_individual_recommendation(top_positive))

        st.markdown("### Visualização individual dos impactos")
        fig, ax = plt.subplots(figsize=(10, 5))

        client_plot_df = (
            client_features.sort_values("abs_shap", ascending=False)
            .head(10)
            .sort_values("shap_value")
        )

        ax.barh(client_plot_df["feature_pt"], client_plot_df["shap_value"])
        ax.set_xlabel("Impacto no risco de churn")
        ax.set_ylabel("Variável")
        ax.set_title("Top fatores do cliente selecionado")

        st.pyplot(fig)
        plt.close(fig)

    # =========================
    # Exploração dos resultados
    # =========================
    st.subheader("🎯 Exploração dos Resultados")

    col_filter_1, col_filter_2 = st.columns(2)

    with col_filter_1:
        selected_risk = st.multiselect(
            "Filtrar por nível de risco",
            options=RISK_ORDER,
            default=RISK_ORDER
        )

    with col_filter_2:
        top_n = st.slider(
            "Quantidade de registros para visualizar",
            min_value=5,
            max_value=min(100, total) if total > 0 else 5,
            value=min(20, total) if total > 0 else 5,
            step=5
        )

    filtered_results = results[results["risk_level"].isin(selected_risk)].copy()

    st.subheader("🔥 Ranking resumido dos clientes")
    ranking_df = filtered_results.copy()
    ranking_df = ranking_df[get_display_columns(ranking_df)]
    ranking_df = translate_for_display(ranking_df)

    if "Probabilidade de Churn (%)" in ranking_df.columns:
        ranking_df["Probabilidade de Churn (%)"] = ranking_df["Probabilidade de Churn (%)"].map(
            lambda x: f"{x:.2f}%"
        )

    st.dataframe(ranking_df.head(top_n), width="stretch")

    with st.expander("📋 Ver resultado completo da base", expanded=False):
        display_results = translate_for_display(filtered_results.copy())

        if "Probabilidade de Churn (%)" in display_results.columns:
            display_results["Probabilidade de Churn (%)"] = display_results["Probabilidade de Churn (%)"].map(
                lambda x: f"{x:.2f}%"
            )

        st.dataframe(display_results, width="stretch")

    csv = filtered_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Baixar resultados em CSV",
        data=csv,
        file_name="churn_resultado.csv",
        mime="text/csv",
    )