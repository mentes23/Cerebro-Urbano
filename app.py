import streamlit as st
from streamlit_chat import message
import pandas as pd
import matplotlib.pyplot as plt
import io
import openai
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Importar configurações do config.py
try:
    from config import OLLAMA_API_BASE, OLLAMA_API_KEY
except ImportError:
    st.error("Erro: O arquivo config.py não foi encontrado. Crie o arquivo config.py na pasta do projeto com as configurações do Ollama.")
    sys.exit(1)

# Configurar o estilo da página (tema escuro, cores do ChatGPT)
st.set_page_config(page_title="Cérebro Urbano", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f5;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .stSelectbox, .stCheckbox, .stDateInput {
            background-color: white;
            border-radius: 8px;
            padding: 5px;
        }
        .stHeader {
            color: #007bff;
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
        }
        .stSubheader {
            color: #333;
            font-size: 1.5em;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Configurar a API do Ollama com deepscaler usando config.py
openai.api_key = OLLAMA_API_KEY
openai.api_base = OLLAMA_API_BASE

# Verificar se o Ollama está rodando
def check_ollama_service():
    try:
        import requests
        response = requests.get(f"{OLLAMA_API_BASE}/api/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.Timeout:
        return False
    except Exception as e:
        print(f"Erro ao verificar Ollama: {str(e)}")
        return False

def get_port_process():
    try:
        import subprocess
        result = subprocess.run(['netstat', '-aon'], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if ':11434' in line:
                parts = line.split()
                if len(parts) >= 5:
                    return parts[-1]
        return None
    except:
        return None

if not check_ollama_service():
    # Tenta encerrar automaticamente qualquer processo na porta 11434
    import check_and_kill_process
    pid = check_and_kill_process.kill_process_on_port(11434)
    if pid:
        st.info(f"Processo {pid} finalizado automaticamente para liberar a porta 11434.")
        import time, subprocess
        time.sleep(2)
    # Tenta iniciar automaticamente o serviço usando a flag de porta
    st.info("Tentando iniciar automaticamente o serviço Ollama na porta 11434...")
    subprocess.Popen(["ollama", "serve", "--port", "11434"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    import time
    time.sleep(5)
    if not check_ollama_service():
        st.error("O serviço Ollama não foi iniciado automaticamente. Verifique se está instalado e configurado corretamente ou inicie manualmente executando 'ollama serve --port 11434'.")
        st.stop()

# Título estilizado
st.markdown("<h1 class='stHeader'>Cérebro Urbano - Assistente Inteligente para Resíduos em Mossoró</h1>", unsafe_allow_html=True)

# Carregar os dados
dados = pd.read_csv("lixo_mossoro.csv")

# Verificar se as colunas esperadas existem
colunas_esperadas = ["dia_semana", "tipo_area", "chuva", "feriado", "quantidade_lixo"]
for col in colunas_esperadas:
    if col not in dados.columns and col != "dia_semana":
        st.error(f"Erro: A coluna '{col}' não está presente no arquivo CSV!")
        st.stop()

# Preparar os dados
if "data" not in dados.columns:
    st.error("A coluna 'data' não foi encontrada no arquivo CSV!")
    st.stop()

try:
    # Converte a coluna 'data' para datetime usando o formato desejado
    dados["data"] = pd.to_datetime(dados["data"], format="%d/%m/%Y")
except Exception as e:
    st.error(f"Erro ao converter a coluna 'data': {e}. Verifique se as datas estão no formato DD/MM/AAAA no arquivo CSV.")
    st.stop()

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
st.markdown(f"<h2 class='stSubheader'>Acurácia do modelo no teste: {acuracia:.2f}</h2>", unsafe_allow_html=True)

# Gráfico histórico de lixo por área (estilizado)
st.markdown("<h2 class='stSubheader'>Histórico de Produção de Lixo por Área</h2>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    dados.groupby("area")["quantidade_lixo"].mean().plot(kind="bar", ax=ax, color=['#007bff', '#28a745', '#dc3545', '#ffc107'])
    ax.set_ylabel("Quantidade Média de Lixo (kg)", fontsize=12)
    ax.set_xlabel("Área", fontsize=12)
    ax.set_title("Produção Média de Lixo por Área", fontsize=14, pad=15)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

# Interface de chatbot completo
st.markdown("<h2 class='stSubheader'>Chat com o Assistente Urbano</h2>", unsafe_allow_html=True)

# Inicializar o estado do chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "message_counter" not in st.session_state:
    st.session_state.message_counter = 0

# Exibir histórico de chat
for msg in st.session_state.messages:
    if "key_id" in msg:
        key = f"msg_{msg['key_id']}"
        if msg["role"] == "user":
            message(msg["content"], is_user=True, avatar_style="adventurer", key=key)
        else:
            message(msg["content"], is_user=False, avatar_style="bottts", key=key)

# Entrada de usuário para o chatbot
user_input = st.text_input("Digite sua pergunta (ex.: 'Quais áreas precisam de mais coletas?')", key="chat_input")

if user_input:
    # Adicionar pergunta do usuário ao chat
    st.session_state.message_counter += 1
    st.session_state.messages.append({"role": "user", "content": user_input, "key_id": st.session_state.message_counter})

    # Chamar a API do Ollama/deepscaler para processar a pergunta
    try:
        response = openai.ChatCompletion.create(
            model="deepscaler",  # Use o modelo deepscaler 1.5B
            messages=[
                {"role": "system", "content": "Você é um assistente inteligente para gestão de resíduos em Mossoró. Use os dados fornecidos para responder perguntas complexas sobre produção de lixo, coletas, e melhorias na infraestrutura urbana. Mantenha respostas curtas e úteis. Se não souber, diga que não tem informações suficientes."},
                *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages if "role" in m and "content" in m]
            ],
            max_tokens=150,
            temperature=0.7
        )

        # Extrair a resposta do deepscaler
        chat_response = response['choices'][0]['message']['content'].strip()

        # Adicionar resposta ao chat
        st.session_state.message_counter += 1
        st.session_state.messages.append({"role": "assistant", "content": chat_response, "key_id": st.session_state.message_counter})

    except Exception as e:
        st.error(f"Erro ao processar a pergunta: {str(e)}. Verifique se o Ollama está rodando na porta 11435 (execute 'ollama serve --port 11435' em outro terminal) e que a porta está livre. Use 'netstat -aon | findstr :11434' para verificar e 'taskkill /F /PID <PID>' para liberar, se necessário.")

    # Atualizar a exibição do chat
    for msg in st.session_state.messages[-2:]:  # Mostrar apenas a última interação
        if "key_id" in msg:
            key = f"msg_{msg['key_id']}"
            if msg["role"] == "user":
                message(msg["content"], is_user=True, avatar_style="adventurer", key=key)
            else:
                message(msg["content"], is_user=False, avatar_style="bottts", key=key)

# Sidebar para configurações de previsões
with st.sidebar:
    st.markdown("<h3 class='stSubheader'>Configurações de Previsões</h3>", unsafe_allow_html=True)
    area_selecionada = st.selectbox("Escolha uma área:", dados["area"].unique(), key="area_select")
    data_inicio = st.date_input("Data de início", value=pd.to_datetime("2025-03-01").date(), key="data_inicio")
    data_fim = st.date_input("Data de fim", value=pd.to_datetime("2025-03-14").date(), key="data_fim", min_value=data_inicio)
    chuva = st.checkbox("Haverá chuva no período?", key="chuva_checkbox")
    feriado = st.checkbox("Haverá feriado no período?", key="feriado_checkbox")

# Preparar os dados para previsão
tipo_area = dados.loc[dados["area"] == area_selecionada, "tipo_area"].iloc[0]
tipo_area_num = le.transform([tipo_area])[0]

# Calcular o número de dias entre as datas
num_dias = (data_fim - data_inicio).days
if num_dias <= 0:
    st.error("A data final deve ser posterior à data inicial!")
    st.stop()

dias = list(range(num_dias))
dias_semana = [pd.to_datetime(data_inicio + pd.Timedelta(days=i)).strftime("%A") for i in range(num_dias)]

futuro = pd.DataFrame({
    "dia_semana": [d % 7 for d in dias],  # Repetir os dias da semana para mais dias, se necessário
    "tipo_area_num": [tipo_area_num] * num_dias,
    "chuva": [1 if chuva else 0] * num_dias,
    "feriado": [1 if feriado else 0] * num_dias
})

# Fazer previsões
previsoes = modelo.predict(futuro)

# Mostrar previsões em formato de chat com chaves únicas
st.session_state.messages.append({"role": "assistant", "content": f"Previsões para {area_selecionada} ({tipo_area}) de {data_inicio} a {data_fim}:", "key_id": st.session_state.message_counter + 1})
st.session_state.message_counter += 1

for i, previsao in enumerate(previsoes):
    st.session_state.message_counter += 1
    dia = dias_semana[i]
    mensagem = f"{dia}: {previsao:.0f} kg"
    st.session_state.messages.append({"role": "assistant", "content": mensagem, "key_id": st.session_state.message_counter})
    if previsao > 700:
        st.session_state.message_counter += 1
        st.session_state.messages.append({"role": "assistant", "content": f"Recomendação: Agendar coleta extra para {dia}!", "style": "warning", "key_id": st.session_state.message_counter})

# Exibir as mensagens de previsão no chat
for msg in st.session_state.messages:
    if "key_id" in msg:
        key = f"msg_{msg['key_id']}"
        if "style" in msg and msg["style"] == "warning":
            message(msg["content"], is_user=False, avatar_style="thumbs", key=key)
        else:
            message(msg["content"], is_user=False, avatar_style="bottts", key=key)

# Botões de exportação na sidebar
if st.sidebar.button("Exportar Previsões como CSV", key="export_button"):
    previsoes_df = pd.DataFrame({
        "Dia": dias_semana,
        "Quantidade (kg)": previsoes.round(0),
        "Coleta Extra": ["Sim" if p > 700 else "Não" for p in previsoes]
    })
    st.sidebar.download_button(
        label="Baixar CSV",
        data=previsoes_df.to_csv(index=False),
        file_name=f"previsoes_{area_selecionada}_{data_inicio}_{data_fim}.csv",
        mime="text/csv",
        key="download_button"
    )

if st.sidebar.button("Exportar Relatório em PDF", key="export_pdf_button"):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    elements = []

    # Estilo para o texto
    styles = getSampleStyleSheet()
    title = Paragraph(f"Relatório de Previsões - {area_selecionada} ({tipo_area})", styles['Heading1'])
    elements.append(title)
    elements.append(Paragraph(f"Período: {data_inicio} a {data_fim}", styles['Normal']))

    # Tabela de previsões
    data = [["Dia", "Quantidade (kg)", "Coleta Extra"]]
    for i, previsao in enumerate(previsoes):
        coleta_extra = "Sim" if previsao > 700 else "Não"
        data.append([dias_semana[i], f"{previsao:.0f}", coleta_extra])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)

    # Gerar o PDF
    doc.build(elements)
    pdf_buffer.seek(0)
    st.sidebar.download_button(
        label="Baixar Relatório em PDF",
        data=pdf_buffer,
        file_name=f"relatorio_{area_selecionada}_{data_inicio}_{data_fim}.pdf",
        mime="application/pdf",
        key="download_pdf_button"
    )