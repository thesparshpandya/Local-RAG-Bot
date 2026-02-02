import os
import sys
import time
import logging
from typing import List

# --- SILENCE CONFIGURATION ---
# Silence LangChain and Unstructured logs
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.ERROR)

# Utility to silence C-level library noise (like PDF "Stroke Color" errors)
class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

# --- Rich Text Imports ---
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.status import Status
    console = Console()
except ImportError:
    print("Tip: Install 'rich' for a better CLI experience (pip install rich)")
    sys.exit(1)

# --- LangChain Imports ---
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# --- CONFIGURATION ---
MODEL_NAME = "mistral"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BASE_URL = "http://localhost:11434"
DB_PATH = "faiss_index"  # Matches your folder structure

def get_files_from_directory(path: str) -> List[str]:
    supported_exts = ['.pdf', '.docx', '.pptx', '.txt']
    files = []
    if os.path.isfile(path): return [path]
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in supported_exts):
                files.append(os.path.join(root, filename))
    return files

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={'normalize_embeddings': True}
    )

def build_vector_store(file_paths: List[str]):
    all_docs = []
    
    with console.status("[bold green]Reading documents (Noise suppressed)...[/bold green]"):
        # We wrap the loader in SuppressStderr to hide the "invalid float" errors
        with SuppressStderr(): 
            for file_path in file_paths:
                try:
                    loader = UnstructuredFileLoader(file_path, strategy="auto")
                    docs = loader.load()
                    all_docs.extend(docs)
                except Exception:
                    pass 

    if not all_docs:
        console.print("[bold red]No documents were successfully loaded.[/bold red]")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    console.print(f"[blue]Processed {len(chunks)} text chunks.[/blue]")

    with console.status("[bold green]Embedding data...[/bold green]"):
        embeddings = get_embedding_model()
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(DB_PATH)
        console.print(f"[green]Brain saved to local folder: '{DB_PATH}'[/green]")
    
    return vector_store

def load_vector_store():
    embeddings = get_embedding_model()
    try:
        vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        console.print(f"[green]Successfully loaded saved brain from '{DB_PATH}'[/green]")
        return vector_store
    except Exception as e:
        console.print(f"[red]Could not load saved data: {e}[/red]")
        return None

def setup_rag_chain(vector_store):
    llm = ChatOllama(model=MODEL_NAME, temperature=0.3, base_url=BASE_URL)

    # --- SMARTER PROMPT ---
    custom_template = """
    You are a smart AI assistant. Follow these rules strictly:
    
    1. If the user says "hi", "hello", "yo", or greets you, simply reply back politely. DO NOT use the context.
    2. If the user asks a question, answer it using ONLY the provided Context.
    3. If the answer is not in the Context, say "I cannot find that information in the documents."
    
    Context: {context}
    Chat History: {chat_history}
    User Input: {question}
    
    Answer:
    """
    
    PROMPT = PromptTemplate(
        input_variables=["context", "chat_history", "question"], 
        template=custom_template
    )

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=False
    )

def main():
    console.clear()
    console.rule("[bold blue]Local RAG Chatbot v2.0 (Persistent)[/bold blue]")
    
    vector_store = None
    
    # Check for existing DB (matches your 'faiss_index' folder)
    if os.path.exists(DB_PATH):
        if Confirm.ask(f"Found saved data in '{DB_PATH}'. Load it?", default=True):
            vector_store = load_vector_store()
    
    if not vector_store:
        # Default matches your 'source_docs' folder
        path_input = Prompt.ask("Enter path to your document or folder", default="source_docs")
        if not os.path.exists(path_input):
            console.print(f"[red]Path not found.[/red]")
            return
        files = get_files_from_directory(path_input)
        vector_store = build_vector_store(files)

    if not vector_store: return

    qa_chain = setup_rag_chain(vector_store)
    console.rule("[bold green]Chat Ready! (Type 'exit' to quit)[/bold green]")

    while True:
        query = Prompt.ask("\n[bold cyan]You[/bold cyan]")
        
        # Check exit BEFORE sending to LLM
        if query.lower() in ["exit", "quit", "q"]: 
            console.print("[yellow]Goodbye![/yellow]")
            break
            
        if not query.strip(): continue

        with console.status("[bold yellow]Thinking...[/bold yellow]"):
            t0 = time.time()
            res = qa_chain.invoke({"question": query})
            t1 = time.time()

        answer = res['answer']
        # Handle sources gracefully (if they exist)
        if 'source_documents' in res:
            sources = list(set([os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in res['source_documents']]))
            source_text = f"Sources: {sources}"
        else:
            source_text = "No sources used (Chitchat)"

        console.print(Panel(Markdown(answer), title="AI Assistant", border_style="green"))
        console.print(f"[dim]{source_text} ({round(t1-t0, 2)}s)[/dim]")

if __name__ == "__main__":
    main()