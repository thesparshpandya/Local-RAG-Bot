import os
import sys

# --- Standard LangChain Imports (Stable v0.3) ---
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# --- The "Chains" Imports ---
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURATION ---
DOCS_FOLDER = "source_docs"
MODEL_NAME = "mistral" 

def main():
    # 1. Validation
    if not os.path.exists(DOCS_FOLDER) or not os.listdir(DOCS_FOLDER):
        print(f"Error: Folder '{DOCS_FOLDER}' is empty or missing.")
        return

    print("Initializing RAG Chatbot (Text-Focused Mode)...")
    
    # 2. Load Documents
    print(f"Loading documents from '{DOCS_FOLDER}'...")
    try:
        # glob="*.pdf" ensures we only try to read PDFs
        # PyPDFLoader ignores complex graphics to prevent errors
        loader = DirectoryLoader(DOCS_FOLDER, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s) successfully.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    # 3. Split Text
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f" -> Created {len(chunks)} text chunks.")

    # 4. Embeddings & Vector Store
    print("Creating vector index (This might take a moment)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 5. Setup the LLM
    llm = ChatOllama(model=MODEL_NAME, temperature=0.1) 

    # 6. Create the Chain
    print("Building logic chain...")
    
    prompt = ChatPromptTemplate.from_template("""
    You are a precise assistant. Answer the question based ONLY on the following context. 
    If the answer is not in the context, say "I don't know based on the document."
    Do not make up facts.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # k=5 means "Look at the top 5 most relevant pages/chunks"
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 7. Chat Loop
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
            
            # Print the Answer
            print(f"\nAI: {response['answer']}\n")
            
            # Show Sources
            print("[Sources Used:]")
            unique_pages = set()
            for doc in response.get("context", []):
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