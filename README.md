# Local RAG Chatbot (Retrieve & Rerank Edition)

A private, local Retrieval-Augmented Generation (RAG) chatbot that allows you to chat with your PDF documents.

This project runs entirely on your local machine using **Ollama** and **LangChain**. No data leaves your computer, ensuring complete privacy. It features a state-of-the-art **Retrieve-and-Rerank** architecture to read complex financial tables, search dense documents, and generate precise, factual answers with source citations.

## 🚀 What's New: The Retrieve-and-Rerank Upgrade

We have overhauled the retrieval pipeline to solve the "Semantic Gap" and "Table Flattening" issues common in standard RAG setups.

### Architecture Evolution: Before vs. After

| Feature | Previous (Naive RAG) | Current (Retrieve & Rerank) |
| --- | --- | --- |
| **Retrieval Engine** | 1-Stage Semantic Search (FAISS direct to LLM) | **2-Stage Pipeline:** Broad Search (FAISS)  Precision Rerank (Cross-Encoder) |
| **Search Breadth** | Narrow (). High risk of missing scattered data. | **Wide Net ():** Instantly grabs a massive pool of candidate chunks. |
| **Precision Filtering** | None. LLM gets flooded with noisy chunks. | **Sniper Precision:** Cross-Encoder reads all 25 chunks word-for-word against the query, passing only the **Top 5** to the LLM. |
| **Chunking Strategy** | Brittle (Size: 1000, Overlap: 200). Split tables apart. | **Massive Overlap (Size: 1000, Overlap: 500):** Guarantees complex tables and their text explanations stay linked. |
| **Table Reading** | Poor. Numbers were easily misattributed. | **Expert Auditor Prompt:** Explicit system guardrails for reading flattened PDF tables accurately. |

## ✨ Key Features

* **100% Local & Private:** Uses local LLMs (Mistral/Llama3) via Ollama; no API keys or cloud costs required.
* **Two-Stage Retrieval (HyDE-Lite + Reranking):**
* **Stage 1 (Bi-Encoder):** `BAAI/bge-small-en-v1.5` grabs the top 25 chunks.
* **Stage 2 (Cross-Encoder):** `BAAI/bge-reranker-base` scores them word-by-word and filters down to the pristine top 5.


* **"Smart" Query Rewriting:**
* **Sticky Context (Coreference Resolution):** Understands that "Calculate it" refers to the previous topic (e.g., "Employee Margin").
* **Semantic Expansion:** Automatically translates user questions (e.g., *"reasons for margin"*) into document-specific language (e.g., *"factors led by..."*).


* **Zero-Latency Startup:** Automatically saves the vector index to disk (`faiss_index`). Subsequent runs load **instantly** (<1s).
* **Logic Guardrails & Honesty:** Handles greetings instantly without DB searches, equates financial synonyms automatically, and explicitly refuses impossible tasks (like counting total words in a PDF).
* **Clean UI:** Features a polished Command Line Interface using the `rich` library.

## 🛠️ Tech Stack

* **Language:** Python 3.11 (Pinned for stability)
* **Orchestration:** LangChain (v0.3 Stable, LCEL Architecture)
* **LLM Engine:** Ollama (Mistral or Llama3)
* **Vector Database:** FAISS (CPU version) with local persistence
* **Embeddings (Bi-Encoder):** `BAAI/bge-small-en-v1.5` (HuggingFace)
* **Reranker (Cross-Encoder):** `BAAI/bge-reranker-base` (Sentence-Transformers)

## 📦 Prerequisites

Before running the project, ensure you have the following installed:

1. **Python 3.11**: This project is optimized for Python 3.11 to ensure compatibility with core AI libraries.
2. **Ollama**: Download from [ollama.com]().
* Once installed, pull the model by running:
```bash
ollama pull mistral

```





## ⚙️ Installation

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



## 🚀 Usage

1. **Prepare your documents:**
Create a folder named `source_docs` in the project root and place your PDF files inside.
2. **Run the Bot:**
```bash
python3 rag_bot.py

```


3. **Chat Interface:**
* **First Run:** The bot will analyze your PDFs, chunk the data, and create a local index. *Note: It will also download the Cross-Encoder model (~1GB) from HuggingFace on the very first query.*
* **Subsequent Runs:** The bot will detect the saved index and ask to load it. Press **Enter** to start instantly.
* **Updating Files:** If you add or remove documents, manually delete the `faiss_index` folder to force a fresh rebuild of the database.
* **To Exit:** Type `exit`, `quit`, or `q`.



## 📁 Project Structure

* `rag_bot.py`: The main application logic. Handles smart rewriting, 2-stage retrieval, reranking, persistence, and the chat loop.
* `source_docs/`: Directory where you store the PDFs you want to chat with.
* `faiss_index/`: (Generated) Stores the vector database locally for speed.
* `requirements.txt`: List of all Python dependencies pinned to stable versions.
* `.gitignore`: Ensures system files and large databases are not uploaded to GitHub.

## 🔧 Troubleshooting

* **"Model not found":** Ensure you have the Ollama app running in the background.
* **Initial Query is Slow:** The very first time you ask a question, the application must download the `bge-reranker-base` model. This is a one-time operation.
* **"Index mismatch" or Bad Answers after changing files:** If the bot behaves unexpectedly after adding new PDFs, manually delete the `faiss_index` folder to force a fresh rebuild with the 1000/500 chunking overlap.
* **PDF Errors:** If a file fails to load, ensure it is a valid PDF (not an empty file or HTML error page) by opening it in a standard PDF viewer first.