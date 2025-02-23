
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, List

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
from collections import defaultdict

class LixoPrevisor:
    """
    Classe para previsão de coleta de lixo com base em dados históricos.

    Attributes:
        dados (pd.DataFrame): DataFrame com os dados históricos de coleta.
        modelo (LinearRegression): Modelo de regressão linear treinado.
        le (LabelEncoder): Encoder para transformar o tipo de área em valores numéricos.
        dias_semana (list): Nomes dos dias da semana.
        areas (list): Lista de áreas de coleta.
    """

    def __init__(self, arquivo_dados: str) -> None:
        """
        Inicializa a classe com o arquivo de dados.

        Args:
            arquivo_dados (str): Caminho para o arquivo CSV com os dados históricos.
        """
        self.dados = pd.read_csv(arquivo_dados, names=['data', 'area', 'quantidade_lixo', 'tipo_area'])
        self.dados['chuva'] = 0
        self.dados['feriado'] = 0
        self.dados["data"] = pd.to_datetime(self.dados["data"], format="%d/%m/%Y")
        self.dados["dia_semana"] = self.dados["data"].dt.dayofweek
        self.le = LabelEncoder()
        self.dados["tipo_area_num"] = self.le.fit_transform(self.dados["tipo_area"])
        self.dias_semana = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
        self.areas = self.dados["area"].unique()
        self._treinar_modelo()

    def _treinar_modelo(self) -> None:
        """
        Treina o modelo de regressão linear.
        """
        X = self.dados[["dia_semana", "tipo_area_num", "chuva", "feriado"]]
        y = self.dados["quantidade_lixo"]
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
        self.modelo = LinearRegression()
        self.modelo.fit(X_treino, y_treino)

        # Avaliação do modelo
        y_previsto = self.modelo.predict(X_teste)
        mse = mean_squared_error(y_teste, y_previsto)
        r2 = r2_score(y_teste, y_previsto)
        print(f"MSE: {mse:.2f}")
        print(f"R²: {r2:.2f}")

    def prever_proxima_semana(self) -> Dict[str, List[float]]:
        """
        Prever a quantidade de lixo para a próxima semana em cada área.

        Returns:
            Dict[str, List[float]]: Dicionário com as previsões para cada área, onde a chave é o nome da área e o valor é uma lista com as previsões para os próximos 7 dias.
        """
        previsoes_semanais = {}
        for area in self.areas:
            tipo_area = self.dados.loc[self.dados["area"] == area, "tipo_area"].iloc[0]
            tipo_area_num = self.le.transform([tipo_area])[0]

            futuro = pd.DataFrame({
                "dia_semana": [0, 1, 2, 3, 4, 5, 6],
                "tipo_area_num": [tipo_area_num] * 7,
                "chuva": [0] * 7,
                "feriado": [0] * 7
            })

            previsoes = self.modelo.predict(futuro)
            previsoes_semanais[area] = previsoes.tolist()
        return previsoes_semanais

    def prever_dias_especificos(self, data_inicio: datetime, dias: int, chuva: int = 0, feriado: int = 0) -> Dict[str, List[float]]:
        """
        Prever a quantidade de lixo para dias específicos.

        Args:
            data_inicio (datetime): Data inicial para a previsão.
            dias (int): Número de dias a serem previstos.
            chuva (int, optional): 0 para sem chuva, 1 para com chuva. Defaults to 0.
            feriado (int, optional): 0 para sem feriado, 1 para com feriado. Defaults to 0.

        Returns:
            Dict[str, List[float]]: Dicionário com as previsões para cada área, onde a chave é o nome da área e o valor é uma lista com as previsões para os dias especificados.
        """
        previsoes_especificas = defaultdict(list)
        for dia in range(dias):
            data_atual = data_inicio + timedelta(days=dia)
            dia_semana = data_atual.weekday()
            for area in self.areas:
                tipo_area = self.dados.loc[self.dados["area"] == area, "tipo_area"].iloc[0]
                tipo_area_num = self.le.transform([tipo_area])[0]

                previsao = self.modelo.predict(pd.DataFrame({
                    "dia_semana": [dia_semana],
                    "tipo_area_num": [tipo_area_num],
                    "chuva": [chuva],
                    "feriado": [feriado]
                }))

                previsoes_especificas[area].append(previsao[0])
        return previsoes_especificas

    def exibir_previsoes(self, previsoes: Dict[str, List[float]], data_inicio: datetime = None, dias: int = None) -> None:
        """
        Exibe as previsões de coleta de lixo.

        Args:
            previsoes (Dict[str, List[float]]): Dicionário com as previsões para cada área.
            data_inicio (datetime, optional): Data inicial para a previsão. Defaults to None.
            dias (int, optional): Número de dias a serem previstos. Defaults to None.
        """
        for area, previsao in previsoes.items():
            print(f"\nPrevisões para {area}:")
            if data_inicio is not None and dias is not None:
                data_atual = data_inicio
                for i, valor in enumerate(previsao):
                    print(f"{data_atual.strftime('%d/%m/%Y')} ({self.dias_semana[data_atual.weekday()]}): {valor:.0f} kg")
                    if valor > 700:
                        print("Recomendação: Agendar coleta extra!")
                    data_atual += timedelta(days=1)
            else:
                for i, valor in enumerate(previsao):
                    print(f"{self.dias_semana[i]}: {valor:.0f} kg")
                    if valor > 700:
                        print("Recomendação: Agendar coleta extra!")

    def simular_interativamente(self) -> None:
        """
        Permite ao usuário simular previsões de coleta de lixo.
        """
        data_inicio_str = input("Digite a data inicial da simulação (dd/mm/aaaa): ")
        try:
            data_inicio = datetime.strptime(data_inicio_str, "%d/%m/%Y")
        except ValueError:
            print("Data inválida. Use o formato dd/mm/aaaa.")
            return

        dias = int(input("Digite o número de dias para a simulação: "))
        chuva = int(input("Haverá chuva na semana? (0 = não, 1 = sim): "))
        feriado = int(input("Haverá feriado na semana? (0 = não, 1 = sim): "))

        previsoes = self.prever_dias_especificos(data_inicio, dias, chuva, feriado)
        self.exibir_previsoes(previsoes, data_inicio, dias)

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
