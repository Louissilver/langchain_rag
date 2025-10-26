# Projeto RAG com LangChain e PMBOK 6ª Edição

Este projeto implementa um sistema de Recuperação Aumentada por Geração (RAG) utilizando a biblioteca LangChain, com foco em responder perguntas sobre o Guia PMBOK 6ª Edição. O sistema carrega um PDF, divide o conteúdo em chunks, indexa os embeddings em uma base vetorial (Chroma) e utiliza um modelo da OpenAI para responder perguntas com base no contexto recuperado e no histórico da conversa.

## Funcionalidades

- Ingestão de documentos PDF e divisão em chunks
- Indexação de embeddings com ChromaDB
- Recuperação de contexto relevante para perguntas
- Geração de respostas com modelo OpenAI (GPT-4o-mini)
- Memória de conversação para contexto contínuo
- Interface interativa via terminal

## Pré-requisitos

- Python 3.8+
- Chave de API da OpenAI (definida na variável de ambiente `OPENAI_API_KEY`)
- Dependências listadas em `requirements.txt`

## Instalação

1. Clone este repositório:
   ```sh
   git clone https://github.com/Louissilver/langchain_rag
   cd projeto_rag
   ```
2. Crie um ambiente virtual (opcional, mas recomendado):
   ```sh
   python -m venv venv
   .venv\Scripts\activate  # Windows
   # ou
   source venv/bin/activate  # Linux/Mac
   ```
3. Instale as dependências:
   ```sh
   pip install -r requirements.txt
   ```
4. Configure a variável de ambiente com sua chave da OpenAI:
   - Crie um arquivo `.env` na raiz do projeto:
     ```env
     OPENAI_API_KEY=sk-...
     ```

## Como usar

1. Coloque o arquivo PDF desejado na pasta `files/` ou informe o caminho completo ao iniciar o script.
2. Execute o script principal:
   ```sh
   python lc_rag.py
   ```
3. Informe o caminho do PDF quando solicitado.
4. Interaja com o assistente digitando perguntas. Digite `sair` para encerrar.

## Estrutura do Projeto

```
projeto_rag/
├── files/                  # PDFs e outros arquivos de entrada
├── lc_rag.py               # Script principal
├── requirements.txt        # Dependências do projeto
└── README.md               # Este arquivo
```

## Principais Tecnologias

- [LangChain](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [OpenAI API](https://platform.openai.com/docs/api-reference)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

## Observações

- O sistema responde apenas com base no conteúdo do PDF fornecido.
- O histórico da conversa é utilizado para melhorar a contextualização das respostas.
- O projeto pode ser adaptado para outros documentos ou domínios, bastando alterar o PDF de entrada.
