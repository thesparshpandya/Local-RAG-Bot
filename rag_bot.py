import os
import sys
import time
import logging
from typing import List

# --- SILENCE CONFIGURATION ---
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.ERROR)

class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    console = Console()
except ImportError:
    print("Tip: Install 'rich' for a better CLI experience")
    sys.exit(1)

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

# --- CONFIGURATION ---
MODEL_NAME = "mistral"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BASE_URL = "http://localhost:11434"
DB_PATH = "faiss_index"

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
    with console.status("[bold green]Reading documents...[/bold green]"):
        with SuppressStderr(): 
            for file_path in file_paths:
                try:
                    loader = UnstructuredFileLoader(file_path, strategy="auto")
                    docs = loader.load()
                    all_docs.extend(docs)
                except Exception: pass 

    if not all_docs:
        console.print("[bold red]No documents loaded.[/bold red]")
        return None

    # Chunk size 1000 ensures specific financial figures aren't cut in half
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    console.print(f"[blue]Processed {len(chunks)} text chunks.[/blue]")

    with console.status("[bold green]Embedding data...[/bold green]"):
        embeddings = get_embedding_model()
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(DB_PATH)
    
    return vector_store

def load_vector_store():
    embeddings = get_embedding_model()
    try:
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception: return None

def setup_rag_chain(vector_store):
    llm = ChatOllama(model=MODEL_NAME, temperature=0, base_url=BASE_URL)

    # Retriever (k=12 covers enough ground for specific financial details)
    retriever = vector_store.as_retriever(search_kwargs={"k": 12})

    # --- THE FIX: QUERY EXPANSION PROMPT ---
    # We now teach the bot that "reasons" == "led by" == "drivers"
    contextualize_q_system_prompt = """You are a smart Search Optimizer.
    Task: Rewrite the user's question to be more effective for searching a financial document.
    
    RULES:
    1. Expand Synonyms: If user asks for "reasons", include words like "drivers", "factors", "led by", "caused by".
       - Example: "Reasons for margin" -> "What factors, drivers, or specific items led to the margin performance?"
    2. Maintain Context: If the user refers to "it" or "this", replace with the specific topic from history.
    3. Simplicity: If the question is simple (e.g., "Who is CEO?"), keep it simple.
    """
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # --- ANSWER PROMPT ---
    qa_system_prompt = """You are an expert Financial Analyst AI.
    Use the provided context to answer the user's question.
    
    STRICT RULES:
    1. Semantics: Understand that "reasons for" usually refers to "what led to" or "drivers of" a metric.
    2. Precision: Quote specific numbers (e.g. "24.6%") if found.
    3. Honesty: If the answer is truly missing, say "I cannot find this information."
    
    Context:
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def main():
    console.clear()
    console.rule("[bold blue]Local RAG Bot v4.1 (Semantic Fix)[/bold blue]")
    
    vector_store = None
    if os.path.exists(DB_PATH) and Confirm.ask(f"Load saved data from '{DB_PATH}'?", default=True):
        vector_store = load_vector_store()
    
    if not vector_store:
        path_input = Prompt.ask("Enter source folder", default="source_docs")
        if not os.path.exists(path_input): return
        files = get_files_from_directory(path_input)
        vector_store = build_vector_store(files)

    if not vector_store: return

    rag_chain = setup_rag_chain(vector_store)
    chat_history = [] 

    console.rule("[bold green]Chat Ready![/bold green]")

    while True:
        query = Prompt.ask("\n[bold cyan]You[/bold cyan]")
        
        # --- GUARDRAIL 1: Instant Exit ---
        if query.lower() in ["exit", "q", "quit"]: 
            console.print("[yellow]Goodbye![/yellow]")
            break
        
        # --- GUARDRAIL 2: Instant Chitchat ---
        greetings = ["hi", "hello", "hey", "yo", "morning", "evening"]
        if query.lower().strip() in greetings:
            console.print(Panel("Hello! How can I help you with your documents today?", title="AI Assistant", border_style="green"))
            continue
            
        if not query.strip(): continue

        with console.status("[bold yellow]Thinking...[/bold yellow]"):
            t0 = time.time()
            result = rag_chain.invoke({"input": query, "chat_history": chat_history})
            t1 = time.time()

        answer = result["answer"]
        sources = list(set([os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in result['context']]))
        
        console.print(Panel(Markdown(answer), title="AI Assistant", border_style="green"))
        console.print(f"[dim]Sources: {sources} ({round(t1-t0, 2)}s)[/dim]")
        
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))
        if len(chat_history) > 6: chat_history = chat_history[-6:]

if __name__ == "__main__":
    main()