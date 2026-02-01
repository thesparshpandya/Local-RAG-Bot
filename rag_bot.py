import os
import json

# --- Standard LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURATION ---
DOCS_FOLDER = "source_docs"
INDEX_FOLDER = "faiss_index"
METADATA_FILE = os.path.join(INDEX_FOLDER, "metadata.json")
MODEL_NAME = "mistral" 

# --- SILENCE WARNINGS (Clean Output) ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )

def get_current_files():
    if not os.path.exists(DOCS_FOLDER):
        return []
    return sorted([f for f in os.listdir(DOCS_FOLDER) if f.endswith('.pdf')])

def create_or_load_vector_store():
    embeddings = get_embeddings()
    current_files = get_current_files()
    
    if not current_files:
        print(f"Error: No PDF files found in '{DOCS_FOLDER}'.")
        return None

    # Smart Sync Check
    if os.path.exists(INDEX_FOLDER) and os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                indexed_files = json.load(f)
            
            if indexed_files == current_files:
                print("Loading existing index (No changes detected)...")
                vector_store = FAISS.load_local(
                    INDEX_FOLDER, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                return vector_store
            else:
                print("Changes detected in file list. Updating index...")
        except Exception:
            print("Index mismatch. Rebuilding...")

    # Rebuild Index with LARGER CHUNKS for better context
    print(f"Loading documents from '{DOCS_FOLDER}'...")
    loader = DirectoryLoader(DOCS_FOLDER, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    print("Splitting text (Optimized for Context)...")
    # FIX 1: Increased chunk size to 1200 to capture full paragraphs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    
    print("Creating vector index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    vector_store.save_local(INDEX_FOLDER)
    with open(METADATA_FILE, "w") as f:
        json.dump(current_files, f)
        
    print("Index saved successfully.")
    return vector_store

def get_rag_chain(vector_store):
    # FIX 2: Temperature 0.2 allows for better synthesis without hallucinations
    llm = ChatOllama(model=MODEL_NAME, temperature=0.2)
    
    # FIX 3: Expert Prompt (Explains instead of just copying)
    prompt = ChatPromptTemplate.from_template("""
    You are an expert analyst. Your goal is to provide a complete and detailed answer based on the provided context.
    
    Instructions:
    1. Read the context below carefully.
    2. Answer the question comprehensively. 
    3. If the answer involves multiple points, list them.
    4. Explain the "Why" and "How" if the text provides it.
    5. Do not simply say "It is X". Say "It is X because..."
    
    If the answer is not in the context, strictly say "I cannot find the answer in the provided documents."

    <context>
    {context}
    </context>

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # FIX 4: Increased k=10 (Looks at more pages to find the needle in the haystack)
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    return create_retrieval_chain(retriever, document_chain)

def main():
    print("Initializing RAG Chatbot...")
    
    vector_store = create_or_load_vector_store()
    if not vector_store:
        return

    retrieval_chain = get_rag_chain(vector_store)

    print("\n" + "="*50)
    print("Chatbot Ready! Type 'exit' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                break
            
            if not query.strip():
                continue

            print("Thinking...", end="\r")
            response = retrieval_chain.invoke({"input": query})
            
            print(f"\nAI: {response['answer']}\n")
            
            print("[Sources Used:]")
            unique_pages = set()
            # Show top 5 sources only to keep UI clean, even though we read 10
            for doc in response.get("context", [])[:5]: 
                page_num = doc.metadata.get('page', 'Unknown')
                src = doc.metadata.get('source', 'Unknown')
                unique_pages.add(f"{os.path.basename(src)} (Page {page_num})")
            
            for p in unique_pages:
                print(f" - {p}")
            print("-" * 50)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()