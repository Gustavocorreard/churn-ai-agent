import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn AI Agent", layout="wide")
st.title("Churn AI Agent")

st.write("✅ App iniciou")
print("✅ App iniciou")

@st.cache_resource
def load_pipeline():
    st.write("🔄 Carregando pipeline...")
    print("🔄 Carregando pipeline...")
    model = joblib.load("models/pipeline.pkl")
    st.write("✅ Pipeline carregado")
    print("✅ Pipeline carregado")
    return model

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Erro ao carregar pipeline: {e}")
    print(f"Erro ao carregar pipeline: {e}")
    st.stop()

uploaded_file = st.file_uploader("Envie um arquivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Arquivo carregado com sucesso")
        st.dataframe(df.head())

        if st.button("Gerar previsões"):
            preds = pipeline.predict(df)
            probs = pipeline.predict_proba(df)[:, 1]

            result = df.copy()
            result["churn_pred"] = preds
            result["churn_prob"] = probs

            st.success("✅ Previsões geradas com sucesso")
            st.dataframe(result.head())

    except Exception as e:
        st.error(f"Erro no processamento: {e}")
        print(f"Erro no processamento: {e}")
