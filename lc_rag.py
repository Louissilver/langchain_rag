import os
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# --- 1. Configuração do Ambiente ---
load_dotenv()
# Certifique-se de que a sua chave da API da OpenAI está configurada como variável de ambiente OPENAI_API_KEY
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não está configurada.")


# --- 2. Preparação dos Dados (Ingestão do PDF) ---
while True:
    pdf_path = input("Informe o caminho do arquivo PDF a ser utilizado: ").strip()
    if os.path.exists(pdf_path):
        break
    print(f"Arquivo PDF não encontrado em: {pdf_path}. Tente novamente.")

print(f"Carregando o documento: {pdf_path}...")
loader = PyPDFLoader(pdf_path)

# Usando Text Splitter para criar chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)  # Ajuste chunk_size/overlap conforme necessário
splits = loader.load_and_split(text_splitter=text_splitter)

if not splits:
    raise ValueError(
        f"Nenhum texto foi extraído ou dividido do PDF: {pdf_path}. Verifique o conteúdo do PDF."
    )

print(f"Documento dividido em {len(splits)} chunks.")


# --- 3. Criação e Armazenamento de Embeddings (Indexação) ---
print("Criando embeddings e indexando no Chroma (isso pode levar um tempo)...")
# Inicializa o modelo de embedding
embeddings_model = OpenAIEmbeddings(
    api_key=openai_api_key, model="text-embedding-3-small"
)

# Cria o Vector Store Chroma em memória a partir dos chunks
# Adiciona um ID único para cada chunk para melhor gerenciamento (opcional, mas bom para atualizações/deleções)
ids = [f"chunk_{i}" for i in range(len(splits))]
vector_store = Chroma.from_documents(
    documents=splits, embedding=embeddings_model, ids=ids
)
print("Indexação concluída.")


# --- 4. Configuração da Recuperação e Geração ---
# Cria um retriever a partir do Vector Store
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)  # Recupera os 3 chunks mais relevantes

# Define o template do prompt para o RAG com histórico
template_with_history = """
Você é um assistente especializado no Guia PMBOK 6ª Edição. Use o histórico da conversa e o contexto a seguir para responder à pergunta atual.
Se você não sabe a resposta com base no contexto ou no histórico, diga explicitamente que a informação não foi encontrada no Guia PMBOK fornecido ou no histórico. Não invente respostas.
Baseie suas respostas *apenas* nas informações do Guia PMBOK 6ª Edição fornecido e no histórico da conversa.
Mantenha a resposta concisa e relevante para o PMBOK.

Histórico da Conversa:
{chat_history}

Contexto Relevante do PMBOK:
{context}

Pergunta Atual: {question}

Resposta útil em Português:
"""
prompt = ChatPromptTemplate.from_template(template_with_history)

# Inicializa o modelo de chat
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)

# Inicializa a memória da conversa - Defina memory_key explicitamente
memory = ConversationBufferMemory(
    memory_key="chat_history",  # Chave para o histórico formatado no prompt
    return_messages=False,
)  # input_key default='input', output_key default='output'


# --- 5. Construção da Chain RAG com LCEL e Memória ---
# Função para formatar os documents recuperados
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Define a chain RAG que aceita 'question' e 'chat_history'
rag_chain_with_history = (
    {
        # Extrai a 'question' do dicionário de entrada e passa para o retriever
        "context": (lambda x: x["question"]) | retriever | format_docs,
        # Passa a pergunta atual diretamente
        "question": lambda x: x["question"],
        # Carrega o histórico da memória usando a memory_key definida
        "chat_history": lambda x: memory.load_memory_variables({})["chat_history"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- 6. Loop Interativo no Terminal ---
print("\n--- Chat Interativo com o Guia PMBOK 6ª Edição ---")
print("Digite 'sair' para terminar a conversa.")

while True:
    try:
        user_input = input("\nVocê: ")
        if user_input.lower() in ["sair", "exit", "quit"]:
            print("IA: Até logo!")
            break

        # Invoca a chain APENAS com a pergunta atual.
        # A chain agora é responsável por carregar o histórico da memória.
        response = rag_chain_with_history.invoke({"question": user_input})

        # Salva a interação atual na memória usando as chaves PADRÃO 'input' e 'output'
        memory.save_context({"input": user_input}, {"output": response})

        print(f"IA: {response}")

    except Exception as e:
        print(f"\nOcorreu um erro: {e}")
        print("Reiniciando o loop ou digite 'sair' para terminar.")
        # Opcional: Limpar a memória em caso de erro grave
        # memory.clear()

# --- Limpeza (opcional) ---
print("\nLimpando o vector store...")
try:
    vector_store.delete_collection()
    print("Vector store limpo.")
except Exception as e:
    print(f"Erro ao limpar o vector store: {e}")
