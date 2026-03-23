import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def remover_duplicatas(df):
    return df.drop_duplicates()

def remover_nulos(df):
    return df.dropna()

def normalizar_coluna(df, column_name, scaler_type="standard"):
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise TypeError(f"A coluna '{column_name}' contém texto ou dados categóricos. Scalers matemáticos exigem dados numéricos.")

    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust": 
        scaler = RobustScaler()
    else: 
        scaler = StandardScaler()
        
    df[column_name] = scaler.fit_transform(df[column_name].values.reshape(-1, 1))
    return df

def renomear_colunas(df, mapeamento: dict):
    return df.rename(columns=mapeamento)

def remover_colunas(df, colunas: list):
    return df.drop(columns=colunas)

def detectar_outliers(df):
    numericas = df.select_dtypes(include="number")
    resultado = {}
    for col in numericas.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        resultado[col] = len(outliers)
    return resultado

def exportar_csv(df, caminho: str):
    df.to_csv(caminho, index=False)

def codificar_categoricas(df, colunas: list):
    return pd.get_dummies(df, columns=colunas)