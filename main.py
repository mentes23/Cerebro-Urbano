import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Carregar os dados do CSV
dados = pd.read_csv("lixo_mossoro.csv", names=['data', 'bairro', 'quantidade_lixo', 'tipo_area'])

# Converter a data para dia da semana (0 = segunda, 6 = domingo)
dados["data"] = pd.to_datetime(dados["data"], format="%d/%m/%Y")
dados["dia_semana"] = dados["data"].dt.dayofweek

# Codificar o tipo de área (ex.: "comercial" vira 0, "residencial" vira 1)
le = LabelEncoder()
dados["tipo_area_num"] = le.fit_transform(dados["tipo_area"])

# Definir variáveis de entrada (X) e saída (y)
X = dados[["dia_semana", "tipo_area_num"]]
y = dados["quantidade_lixo"]

# Dividir os dados em treino (80%) e teste (20%)
X_treino, X_teste, y_treino, y_teste = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

# Criar e treinar o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X_treino, y_treino)

# Mostrar a acurácia do modelo
print("Acurácia no teste:", modelo.score(X_teste, y_teste))

# Prever o lixo para os próximos 7 dias na área comercial
futuro = pd.DataFrame({
    "dia_semana": [0, 1, 2, 3, 4, 5, 6],  # Segunda a domingo
    "tipo_area_num": [le.transform(["comercial"])[0]] * 7
})
previsoes = modelo.predict(futuro)

# Exibir previsões e recomendações
for dia, previsao in enumerate(previsoes):
    print(f"Dia {dia} (segunda a domingo): {previsao:.0f} kg")
    if previsao > 700:
        print("Recomendação: Agendar coleta extra!")
