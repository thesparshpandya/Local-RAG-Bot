# Local RAG Chatbot

A private, local Retrieval-Augmented Generation (RAG) chatbot that allows you to chat with your PDF documents.

This project runs entirely on your local machine using **Ollama** and **LangChain**. No data leaves your computer, ensuring complete privacy. It uses vector embeddings to search your documents and a local LLM to generate precise, factual answers with source citations.

## Key Features

* **100% Local & Private:** Uses local LLMs (Mistral) via Ollama; no API keys or cloud costs required.
* **PDF Support:** robust parsing using `PyPDFLoader` to handle text extraction reliably.
* **Source Citations:** Every answer includes the exact filename and page number where the information was found.
* **Vector Search:** Uses FAISS and HuggingFace embeddings for semantic search (finding meaning, not just keywords).
* **Fact-Focused:** Tuned for low temperature (0.1) to reduce hallucinations and stick strictly to the provided context.

## Tech Stack

* **Language:** Python 3.11 (Pinned for stability)
* **Orchestration:** LangChain (v0.3 Stable Architecture)
* **LLM Engine:** Ollama (running Mistral)
* **Vector Database:** FAISS (CPU version)
* **Embeddings:** `BAAI/bge-small-en-v1.5` (HuggingFace)

## Prerequisites

Before running the project, ensure you have the following installed:

1. **Python 3.11**: This project is optimized for Python 3.11 to ensure compatibility with core AI libraries.
2. **Ollama**: Download from [ollama.com](https://ollama.com).
* Once installed, pull the model by running:
```bash
ollama pull mistral

```





## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Local-RAG-Bot.git
cd Local-RAG-Bot

```


2. **Create a Virtual Environment (Recommended):**
It is best to use a fresh environment to avoid conflicts.
```bash
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate

```


*(Note: Adjust the python path if your installation is located elsewhere)*
3. **Install Dependencies:**
```bash
pip install -r requirements.txt

```



## Usage

1. **Prepare your documents:**
Create a folder named `source_docs` in the project root and place your PDF files inside.
```text
Local-RAG-Bot/
├── source_docs/
│   ├── manual.pdf
│   └── report.pdf

```


2. **Run the Bot:**
```bash
python3 rag_bot.py

```


3. **Chat:**
The bot will index your documents (this happens once per run) and then prompt you for input. Type your question and hit Enter. Type `exit` to quit.

## Project Structure

* `rag_bot.py`: The main application logic. Handles document loading, splitting, embedding, and the chat loop.
* `requirements.txt`: List of all Python dependencies pinned to stable versions.
* `source_docs/`: Directory where you store the PDFs you want to chat with.
* `.gitignore`: Specifies files that should be ignored by Git (like the `venv` folder and `.DS_Store`).

## Troubleshooting

* **"Model not found":** Ensure you have the Ollama app running in the background and that you have run `ollama pull mistral`.
* **"Folder is empty":** Make sure you created the `source_docs` folder and added at least one PDF file.
* **Slow performance:** The first time you run a query, the system might be slow as it loads the embedding model into memory. Subsequent queries will be faster. Performance depends on your hardware (CPU/RAM).