import pandas as pd

REQUIRED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

OPTIONAL_ID_COLUMNS = [
    "customerID",
    "CustomerID",
    "customer_id",
    "Unnamed: 0",
    "Churn",
]

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

COLUMN_ALIASES = {
    "Gender": "gender",
    "Tenure": "tenure",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges": "TotalCharges",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = df.rename(columns=COLUMN_ALIASES)
    return df


def strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    return df


def convert_senior_citizen(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "SeniorCitizen" in df.columns:
        mapping = {
            "Yes": 1,
            "No": 0,
            "yes": 1,
            "no": 0,
            "Y": 1,
            "N": 0,
            "True": 1,
            "False": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            1: 1,
            0: 0,
            True: 1,
            False: 0,
        }

        df["SeniorCitizen"] = df["SeniorCitizen"].map(lambda x: mapping.get(x, x))
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")

    return df


def convert_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = (
            df["TotalCharges"]
            .astype(str)
            .str.strip()
            .str.replace(",", ".", regex=False)
        )

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def convert_numeric_columns(df: pd.DataFrame):
    df = df.copy()
    type_errors = []

    for col in NUMERIC_COLUMNS:
        if col in df.columns and col != "TotalCharges":
            df[col] = pd.to_numeric(df[col], errors="coerce")

            invalid_count = df[col].isna().sum()
            if invalid_count > 0:
                type_errors.append(
                    f"A coluna '{col}' possui {invalid_count} valor(es) inválido(s) ou vazio(s)."
                )

    return df, type_errors


def impute_missing_values(df: pd.DataFrame):
    df = df.copy()
    warnings = []

    if "TotalCharges" in df.columns:
        missing_total_charges = df["TotalCharges"].isna().sum()

        if missing_total_charges > 0:
            median_value = df["TotalCharges"].median()

            if pd.isna(median_value):
                median_value = 0

            df["TotalCharges"] = df["TotalCharges"].fillna(median_value)

            warnings.append(
                f"A coluna 'TotalCharges' tinha {missing_total_charges} valor(es) nulo(s) e foi preenchida com a mediana ({median_value:.2f})."
            )

    return df, warnings


def validate_input_data(df: pd.DataFrame) -> dict:
    df = normalize_columns(df)
    df = strip_string_values(df)
    df = convert_senior_citizen(df)
    df = convert_total_charges(df)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    extra_columns = [
        col for col in df.columns
        if col not in REQUIRED_COLUMNS and col not in OPTIONAL_ID_COLUMNS
    ]

    df, type_errors = convert_numeric_columns(df)
    df, imputation_warnings = impute_missing_values(df)

    valid = len(missing_columns) == 0 and len(type_errors) == 0

    return {
        "valid": valid,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "type_errors": type_errors,
        "warnings": imputation_warnings,
        "normalized_df": df,
    }