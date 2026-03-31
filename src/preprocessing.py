import pandas as pd


def preprocess_data(df):
    
    # =========================
    # 1. Remover colunas inúteis
    # =========================
    drop_cols = ['customerID', 'Unnamed: 0']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # =========================
    # 2. Converter variável alvo
    # =========================
    df['Churn'] = df['Churn'].map({'Churned': 1, 'Stayed': 0})
    
    # =========================
    # 3. Corrigir coluna numérica
    # =========================
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # =========================
    # 4. Converter Yes/No para 1/0 (otimiza modelo)
    # =========================
    binary_map = {'Yes': 1, 'No': 0}
    
    for col in df.columns:
        if df[col].dtype == 'object':
            if set(df[col].unique()).issubset(set(['Yes', 'No'])):
                df[col] = df[col].map(binary_map)
    
    # =========================
    # 5. Encoding categórico
    # =========================
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df