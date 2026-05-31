# OmniRAG Engine

**An end-to-end, local Retrieval-Augmented Generation (RAG) system built with LangChain, Faiss, and Gemma2, featuring a multi-modal data ingestion pipeline for semantic search and summarization.**

## 🚀 Features

- **100% Local Inference**: Uses [Ollama](https://ollama.com/) with the `gemma2:2b` model to ensure complete data privacy and offline capability.
- **High-Performance Vector Search**: Leverages `faiss-cpu` for lightning-fast similarity search of high-dimensional embeddings.
- **Omni-Format Data Ingestion**: Built-in support for parsing and chunking various document types:
  - PDF (`PyPDFLoader`)
  - Text files (`TextLoader`)
  - CSV (`CSVLoader`)
  - Excel (`UnstructuredExcelLoader`)
  - Word Documents (`Docx2txtLoader`)
  - JSON (`JSONLoader`)
- **Semantic Embeddings**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to generate rich contextual embeddings.

## 🛠️ Architecture & Tech Stack

- **Framework**: [LangChain](https://python.langchain.com/)
- **LLM**: Gemma 2 (via Ollama)
- **Vector Store**: Faiss (Facebook AI Similarity Search)
- **Embeddings**: Sentence-Transformers
- **Dependency Management**: `uv`

## 📁 Project Structure

```text
ProjectRAG/
├── data/                  # Drop your documents (PDF, TXT, CSV, Excel, Word, JSON) here
├── faiss_store/           # Automatically generated persistent vector database
├── src/
│   ├── data_loader.py     # Multi-format document ingestion pipeline
│   ├── embedding.py       # Text chunking and embedding generation
│   ├── search.py          # RAG search logic and LLM prompt formulation
│   └── vectorstore.py     # Faiss index management
├── app.py                 # Main entry point to run the RAG pipeline
├── pyproject.toml         # Dependencies and project metadata
└── README.md
```

## ⚙️ Setup & Installation

### Prerequisites
1. **Python 3.13+**
2. **[uv](https://github.com/astral-sh/uv)** (Lightning-fast Python package manager)
3. **[Ollama](https://ollama.com/)** (To run the local LLM)

### 1. Install Dependencies
Navigate to the project root and install the required dependencies using `uv`:

```bash
uv sync
```

### 2. Download the LLM Model
Ensure you have Ollama installed and running in the background. Then, pull the required model:

```bash
ollama run gemma2:2b
```

## 🏃‍♂️ Usage

1. **Add your documents**: Place any supported files (PDF, Word, TXT, CSV, Excel, JSON) into the `data/` directory.
2. **Run the application**: Execute `app.py` from the root of the project to process the documents, build the vector store, and execute a query.

```bash
uv run python app.py
```
*(Or, if your virtual environment is already activated, simply run `python app.py`)*

### Customizing the Query
Open `app.py` and modify the `query` variable to ask questions based on your custom documents:

```python
rag_search = RAGSearch()
query = "What are biomaterials?"
summary = rag_search.search_and_summarize(query, top_k=5)
print("Summary:", summary)
```

