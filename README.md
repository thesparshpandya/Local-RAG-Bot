# Local RAG Chatbot

A private, local Retrieval-Augmented Generation (RAG) chatbot that allows you to chat with your PDF documents.

This project runs entirely on your local machine using **Ollama** and **LangChain**. No data leaves your computer, ensuring complete privacy. It uses vector embeddings to search your documents and a local LLM to generate precise, factual answers with source citations.

## Key Features

* **100% Local & Private:** Uses local LLMs (Mistral/Llama3) via Ollama; no API keys or cloud costs required.
* **Smart Sync & Persistence:** Automatically saves the vector index to disk (`faiss_index`). Subsequent runs load **instantly** (<1s) unless you add/remove files.
* **Expert Analysis:** Uses a tuned "Expert Persona" prompt to provide detailed explanations rather than just retrieving raw text.
* **Deep Context:** Scans the top 10 relevant document chunks with optimized slicing (1200 characters) to capture full paragraphs and context.
* **Cross-Platform:** Optimized file handling for macOS, Windows, and Linux.

## Tech Stack

* **Language:** Python 3.11 (Pinned for stability)
* **Orchestration:** LangChain (v0.3 Stable Architecture)
* **LLM Engine:** Ollama (Mistral or Llama3)
* **Vector Database:** FAISS (CPU version) with local persistence
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
```bash
python3.11 -m venv venv
source venv/bin/activate
# On Windows use: venv\Scripts\activate

```


3. **Install Dependencies:**
```bash
pip install -r requirements.txt

```



## Usage

1. **Prepare your documents:**
Create a folder named `source_docs` in the project root and place your PDF files inside.
2. **Run the Bot:**
```bash
python3 rag_bot.py

```


3. **Chat:**
* **First Run:** The bot will analyze your PDFs and create a local index (takes ~30-60s depending on file size).
* **Subsequent Runs:** The bot will load the saved index **instantly**.
* **Updating Files:** If you add or remove PDFs, the bot detects the change automatically and rebuilds the index.



## Project Structure

* `rag_bot.py`: The main application logic. Handles document loading, splitting, persistence, and the chat loop.
* `source_docs/`: Directory where you store the PDFs you want to chat with.
* `faiss_index/`: (Generated) Stores the vector database locally for speed.
* `requirements.txt`: List of all Python dependencies pinned to stable versions.
* `.gitignore`: Ensures system files and large databases are not uploaded to GitHub.

## Troubleshooting

* **"Model not found":** Ensure you have the Ollama app running in the background.
* **"Index mismatch":** If the bot behaves unexpectedly, you can manually delete the `faiss_index` folder to force a fresh rebuild.
* **PDF Errors:** If a file fails to load, ensure it is a valid PDF (not an empty file or HTML error page) by opening it in a standard PDF viewer first.