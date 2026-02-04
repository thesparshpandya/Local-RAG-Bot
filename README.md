# Local RAG Chatbot

A private, local Retrieval-Augmented Generation (RAG) chatbot that allows you to chat with your PDF documents.

This project runs entirely on your local machine using **Ollama** and **LangChain**. No data leaves your computer, ensuring complete privacy. It uses vector embeddings to search your documents and a local LLM to generate precise, factual answers with source citations.

## Key Features

* **100% Local & Private:** Uses local LLMs (Mistral/Llama3) via Ollama; no API keys or cloud costs required.
* **Zero-Latency Startup:** Automatically saves the vector index to disk (`faiss_index`). Subsequent runs load **instantly** (<1s) unless you add/remove files.
* **"Smart" Query Rewriting:**
    * **Coreference Resolution:** Understands that "it" or "this project" refers to the previous topic (e.g., "BSNL deal").
    * **Semantic Expansion:** Automatically translates user questions (e.g., *"reasons for margin"*) into document-specific language (e.g., *"factors led by..."*), fixing the common "Semantic Gap" in RAG.
    * **Loop Prevention:** Intelligently detects when you change the topic to stop the bot from getting stuck in previous contexts.
* **Deep Context (k=12):** Scans the top 12 relevant document chunks with optimized slicing (1000 characters) to find specific details (like CEO names) buried deep in long reports without hallucinating.
* **Logic Guardrails:**
    * **Instant Chitchat:** Handles greetings ("hi", "yo") instantly without triggering a database search.
    * **Financial Synonyms:** Automatically treats "Operating Margin" and "EBIT Margin" as interchangeable.
    * **Honesty Checks:** Explicitly admits when it cannot perform global tasks (like counting specific words across an entire document).
* **Smart OCR Strategy:** Uses `Unstructured` with `strategy="auto"` to automatically switch between fast text extraction and OCR for scanned images/tables.
* **Clean UI:** Features a polished Command Line Interface using the `rich` library, with automatic suppression of noisy PDF parsing errors.

## Tech Stack

* **Language:** Python 3.11 (Pinned for stability)
* **Orchestration:** LangChain (LCEL Architecture)
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
git clone https://github.com/thesparshpandya/Local-RAG-Bot.git
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
* **Subsequent Runs:** The bot will detect the saved index and ask to load it. Press **Enter** to start instantly.
* **Updating Files:** If you add new documents, delete the `faiss_index` folder to force a rebuild.
* **To Exit:** Type `exit`, `quit`, or `q`.

## Project Structure

* `rag_bot.py`: The main application logic. Handles smart rewriting, retrieval, persistence, and the chat loop.
* `source_docs/`: Directory where you store the PDFs you want to chat with.
* `faiss_index/`: (Generated) Stores the vector database locally for speed.
* `requirements.txt`: List of all Python dependencies pinned to stable versions.
* `.gitignore`: Ensures system files and large databases are not uploaded to GitHub.

## Troubleshooting

* **"Model not found":** Ensure you have the Ollama app running in the background.
* **Bot answers "I cannot find this information":** Try being more specific. While the bot has semantic expansion, extremely vague queries on large documents may still miss context.
* **"Index mismatch":** If the bot behaves unexpectedly after changing files, manually delete the `faiss_index` folder to force a fresh rebuild.
* **PDF Errors:** If a file fails to load, ensure it is a valid PDF (not an empty file or HTML error page) by opening it in a standard PDF viewer first.

```

```