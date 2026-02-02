import os
import sys
import time
import warnings
from typing import List

# Filter warnings to keep terminal clean
warnings.filterwarnings("ignore")

# --- Rich Text Imports (For structured terminal output) ---
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.status import Status
    console = Console()
except ImportError:
    # Fallback if rich is not installed
    print("Tip: Install 'rich' for a better CLI experience (pip install rich)")
    class Console:
        def print(self, *args, **kwargs): print(*args)
        def rule(self, *args): print("-" * 50)
        def clear(self): os.system('cls' if os.name == 'nt' else 'clear')
    class Markdown:
        def __init__(self, text): self.text = text
        def __str__(self): return self.text
    class Panel:
        def __init__(self, renderable, title="", **kwargs): self.renderable = renderable
    class Prompt:
        @staticmethod
        def ask(text, default=""): return input(f"{text} [{default}]: ") or default
    class Status:
        def __init__(self, text): print(f"[{text}]...")
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    console = Console()

# --- LangChain & AI Imports ---
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# --- CONFIGURATION ---
MODEL_NAME = "mistral"  # LLM Model Name
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BASE_URL = "http://localhost:11434"

def get_files_from_directory(path: str) -> List[str]:
    """Scans a directory for supported document files."""
    supported_exts = ['.pdf', '.docx', '.pptx', '.txt']
    files = []
    
    # Handle single file path
    if os.path.isfile(path):
        return [path]
    
    # Handle directory path
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in supported_exts):
                files.append(os.path.join(root, filename))
    return files

def process_documents(file_paths: List[str]):
    """
    Loads, splits, and embeds documents.
    Uses 'strategy=auto' to handle both text PDFs and scanned images (OCR).
    """
    all_docs = []
    
    # 1. Load Documents
    with console.status("[bold green]Reading documents...[/bold green]") as status:
        for i, file_path in enumerate(file_paths):
            console.print(f"[dim]Processing: {os.path.basename(file_path)}[/dim]")
            try:
                # strategy="auto" detects if OCR (Tesseract) is needed
                loader = UnstructuredFileLoader(file_path, strategy="auto")
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                console.print(f"[bold red]Error loading {os.path.basename(file_path)}: {e}[/bold red]")

    if not all_docs:
        console.print("[bold red]No documents were successfully loaded.[/bold red]")
        return None

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)
    console.print(f"[blue]Created {len(chunks)} text chunks.[/blue]")

    # 3. Create Vector Store
    with console.status("[bold green]Creating Vector Database (Embeddings)...[/bold green]"):
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={'normalize_embeddings': True}
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

def setup_rag_chain(vector_store):
    """
    Sets up the conversational chain with a custom persona and memory.
    """
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.3, # Keep it factual
        base_url=BASE_URL
    )

    # Custom System Prompt (The "Persona")
    custom_template = """
    You are a highly skilled Document Analysis AI. Your goal is to provide accurate, 
    professional answers based ONLY on the provided context.
    
    If the answer is not in the context, say "I cannot find that information in the documents."
    Do not make up facts.
    
    Context: {context}
    
    Chat History: {chat_history}
    User Question: {question}
    
    Answer:
    """
    
    PROMPT = PromptTemplate(
        input_variables=["context", "chat_history", "question"], 
        template=custom_template
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}), # Retrieve top 4 chunks
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=False
    )
    
    return chain

def main():
    console.clear()
    console.rule("[bold blue]Local RAG Chatbot (CLI Version)[/bold blue]")
    
    # 1. Get Document Path
    # Default is now set to 'source_docs' as requested
    path_input = Prompt.ask("Enter path to your document or folder", default="source_docs")
    
    if not os.path.exists(path_input):
        console.print(f"[bold red]Error: Path '{path_input}' does not exist.[/bold red]")
        return

    # 2. Find Files
    files = get_files_from_directory(path_input)
    if not files:
        console.print("[bold red]No PDF/DOCX/TXT files found in that location.[/bold red]")
        return
    
    console.print(f"[green]Found {len(files)} documents.[/green]")
    
    # 3. Build Brain
    vector_store = process_documents(files)
    if not vector_store:
        return

    # 4. Setup Chain
    qa_chain = setup_rag_chain(vector_store)
    console.rule("[bold green]Chat Ready! (Type 'exit' to quit)[/bold green]")

    # 5. Chat Loop
    while True:
        try:
            query = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            if query.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not query.strip():
                continue

            with console.status("[bold yellow]Thinking...[/bold yellow]"):
                start_time = time.time()
                response = qa_chain.invoke({"question": query})
                end_time = time.time()

            # 6. Display Answer
            answer = response['answer']
            sources = response['source_documents']
            
            # Print Markdown Answer
            console.print(Panel(Markdown(answer), title="[bold green]AI Assistant[/bold green]", border_style="green"))
            
            # Print Sources (Optional but useful for debugging)
            unique_sources = set(doc.metadata.get('source', 'Unknown') for doc in sources)
            console.print(f"[dim]Sources: {', '.join([os.path.basename(s) for s in unique_sources])} ({round(end_time - start_time, 2)}s)[/dim]")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            console.print(f"[bold red]An error occurred: {e}[/bold red]")

if __name__ == "__main__":
    main()