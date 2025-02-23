
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Carregar os dados
dados = pd.read_csv("lixo_mossoro.csv", names=['data', 'area', 'quantidade_lixo', 'tipo_area'])

# Adicionar colunas faltantes
dados['chuva'] = 0
dados['feriado'] = 0

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
print("Acurácia no teste:", modelo.score(X_teste, y_teste))

# Definir variáveis globais
dias_semana = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
areas = dados["area"].unique()  # Pegar todas as áreas do CSV

# Prever para os próximos 7 dias para cada área
for area in areas:
    tipo_area = dados.loc[dados["area"] == area, "tipo_area"].iloc[0]
    tipo_area_num = le.transform([tipo_area])[0]

    # Simular uma semana sem chuva e sem feriado
    futuro = pd.DataFrame({
        "dia_semana": [0, 1, 2, 3, 4, 5, 6],  # Segunda a domingo
        "tipo_area_num": [tipo_area_num] * 7,
        "chuva": [0] * 7,  # Sem chuva
        "feriado": [0] * 7  # Sem feriado
    })

    previsoes = modelo.predict(futuro)
    print(f"\nPrevisões para {area} (tipo: {tipo_area}):")
    for i, previsao in enumerate(previsoes):
        print(f"{dias_semana[i]}: {previsao:.0f} kg")
        if previsao > 700:
            print("Recomendação: Agendar coleta extra!")

# Simulação interativa
print("\nSimulação de previsões:")
chuva = int(input("Haverá chuva na semana? (0 = não, 1 = sim): "))
feriado = int(input("Haverá feriado na semana? (0 = não, 1 = sim): "))

for area in areas:
    tipo_area = dados.loc[dados["area"] == area, "tipo_area"].iloc[0]
    tipo_area_num = le.transform([tipo_area])[0]
    
    futuro = pd.DataFrame({
        "dia_semana": [0, 1, 2, 3, 4, 5, 6],
        "tipo_area_num": [tipo_area_num] * 7,
        "chuva": [chuva] * 7,
        "feriado": [feriado] * 7
    })
    
    previsoes = modelo.predict(futuro)
    print(f"\nPrevisões para {area} (tipo: {tipo_area}):")
    for i, previsao in enumerate(previsoes):
        print(f"{dias_semana[i]}: {previsao:.0f} kg")
        if previsao > 700:
            print("Recomendação: Agendar coleta extra!")
