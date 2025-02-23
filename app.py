import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Título da aplicação
st.title("Cérebro Urbano - Gestão Inteligente de Resíduos em Mossoró")

# Carregar os dados
dados = pd.read_csv("lixo_mossoro.csv")

# Verificar se as colunas esperadas existem
colunas_esperadas = ["dia_semana", "tipo_area", "chuva", "feriado", "quantidade_lixo"]
for col in colunas_esperadas:
    if col not in dados.columns and col != "dia_semana":  # dia_semana é criado depois
        st.error(f"Erro: A coluna '{col}' não está presente no arquivo CSV!")
        st.stop()

# Converter a data para dia da semana
dados["data"] = pd.to_datetime(dados["data"], format="%d/%m/%Y")
dados["dia_semana"] = dados["data"].dt.dayofweek

# Codificar o tipo de área
le = LabelEncoder()
dados["tipo_area_num"] = le.fit_transform(dados["tipo_area"])

# Definir variáveis de entrada (X) e saída (y)
X = dados[["dia_semana", "tipo_area_num", "chuva", "feriado"]]
y = dados["quantidade_lixo"]

# Dividir os dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(X_treino, y_treino)

# Mostrar a acurácia
acuracia = modelo.score(X_teste, y_teste)
st.write(f"**Acurácia do modelo no teste:** {acuracia:.2f}")

# Gráfico histórico de lixo por área
st.subheader("Histórico de Produção de Lixo por Área")
fig, ax = plt.subplots()
dados.groupby("area")["quantidade_lixo"].mean().plot(kind="bar", ax=ax)
ax.set_ylabel("Quantidade Média de Lixo (kg)")
ax.set_xlabel("Área")
st.pyplot(fig)

# Interface interativa para previsões
st.subheader("Previsões para os Próximos 7 Dias")
area_selecionada = st.selectbox("Escolha uma área:", dados["area"].unique())
chuva = st.checkbox("Haverá chuva na semana?")
feriado = st.checkbox("Haverá feriado na semana?")

# Preparar os dados para previsão
tipo_area = dados.loc[dados["area"] == area_selecionada, "tipo_area"].iloc[0]
tipo_area_num = le.transform([tipo_area])[0]
dias_semana = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]

futuro = pd.DataFrame({
    "dia_semana": [0, 1, 2, 3, 4, 5, 6],
    "tipo_area_num": [tipo_area_num] * 7,
    "chuva": [1 if chuva else 0] * 7,
    "feriado": [1 if feriado else 0] * 7
})

# Fazer previsões
previsoes = modelo.predict(futuro)

# Mostrar resultados
st.write(f"**Previsões para {area_selecionada} (tipo: {tipo_area}):**")
for i, previsao in enumerate(previsoes):
    st.write(f"{dias_semana[i]}: {previsao:.0f} kg")
    if previsao > 700:
        st.warning(f"Recomendação: Agendar coleta extra para {dias_semana[i]}!")

# Botão para exportar previsões
if st.button("Exportar Previsões como CSV"):
    previsoes_df = pd.DataFrame({
        "Dia": dias_semana,
        "Quantidade (kg)": previsoes.round(0),
        "Coleta Extra": ["Sim" if p > 700 else "Não" for p in previsoes]
    })
    st.download_button(
        label="Baixar CSV",
        data=previsoes_df.to_csv(index=False),
        file_name=f"previsoes_{area_selecionada}.csv",
        mime="text/csv"
    )