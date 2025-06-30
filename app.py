import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(layout="wide")
st.markdown("# Dashboard de Análise PETR4")

# Carregamento dos dados
@st.cache_data
def carregar_dados():
    return pd.read_csv("petr4.csv")

# Preparo dos dados
def preparar_dados(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Target"] = np.where(df["Close"] > df["Open"], 1, 0)
    df["Year"] = df["Date"].dt.year
    df["Profit"] = np.where(df["Target"] == 1, df["Close"] - df["Open"], df["Open"] - df["Close"])
    return df

# Métricas
def calcular_metricas(y_true, y_pred):
    return {
        "Acurácia": accuracy_score(y_true, y_pred),
        "Precisão": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "Especificidade": confusion_matrix(y_true, y_pred)[0, 0] / sum(confusion_matrix(y_true, y_pred)[0])
    }

# Simulação de retorno
def simular_retorno(y_true, y_pred, lucros):
    acertos = (y_true == y_pred)
    retorno_ganhos = lucros[acertos & (y_true == 1)].sum()
    retorno_perdas = lucros[~acertos & (y_true == 1)].sum()
    retorno_total = retorno_ganhos - retorno_perdas
    return retorno_ganhos, retorno_perdas, retorno_total

dados = carregar_dados()
dados = preparar_dados(dados)

# Seleção de anos de treino
anos_treino = st.sidebar.multiselect("Selecione os anos para Treinamento", sorted(dados["Year"].unique()), default=[2023])
df_treino = dados[dados["Year"].isin(anos_treino)]
df_teste = dados[~dados["Year"].isin(anos_treino)]

# Normalização
def normalizar(df):
    return (df - df.mean()) / df.std()

X_treino = normalizar(df_treino[["Open", "High", "Low", "Close", "Volume"]])
y_treino = df_treino["Target"]

X_teste = normalizar(df_teste[["Open", "High", "Low", "Close", "Volume"]])
y_teste = df_teste["Target"]

# Modelo
modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(X_treino, y_treino)
y_pred = modelo.predict(X_teste)

# Avaliações Métricas
st.subheader("Avaliações Métricas")
metricas = calcular_metricas(y_teste, y_pred)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Acurácia", f"{metricas['Acurácia']*100:.2f}%")
col2.metric("Precisão", f"{metricas['Precisão']*100:.2f}%")
col3.metric("Recall (Sens.)", f"{metricas['Recall']*100:.2f}%")
col4.metric("F1-Score", f"{metricas['F1-Score']*100:.2f}%")
col5.metric("Especificidade", f"{metricas['Especificidade']*100:.2f}%")

# Retorno Financeiro
lucros_teste = df_teste["Profit"].values
ganhos, perdas, total = simular_retorno(y_teste.values, y_pred, lucros_teste)
st.subheader("Retorno Financeiro")
col6, col7, col8 = st.columns(3)
col6.metric("Retorno de Ganhos", f"R$ {ganhos:.2f}")
col7.metric("Retorno de Perdas", f"R$ {perdas:.2f}")
col8.metric("Retorno Total", f"R$ {total:.2f}")

# Gráficos
st.subheader("Gráficos Interativos")
grafico = st.selectbox("Selecione o gráfico", ["Série Temporal Completa", "Distribuição da Variável Alvo"])

if grafico == "Série Temporal Completa":
    fig = px.line(dados, x="Date", y="Close", color="Year", title="Preço de Fechamento - PETR4")
    st.plotly_chart(fig, use_container_width=True)

elif grafico == "Distribuição da Variável Alvo":
    dist = dados["Target"].value_counts(normalize=True) * 100
    fig = px.bar(
        x=["Baixa (0)", "Alta (1)"],
        y=dist.values,
        labels={"x": "Classe", "y": "Percentual"},
        title="Distribuição da Variável Alvo em %",
        color=["Baixa", "Alta"],
        color_discrete_sequence=["red", "green"]
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**Alta:** {dist[1]:.2f}% | **Baixa:** {dist[0]:.2f}%")
