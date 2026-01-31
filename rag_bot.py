import os
import sys

# --- Standard LangChain Imports (Stable v0.3) ---
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# --- The "Chains" Imports ---
# These specific paths work perfectly with the pinned versions in requirements.txt
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURATION ---
DOCS_FOLDER = "source_docs"
MODEL_NAME = "mistral"

def main():
    # 1. Validation
    if not os.path.exists(DOCS_FOLDER) or not os.listdir(DOCS_FOLDER):
        print(f"Error: Folder '{DOCS_FOLDER}' is empty.")
        return

    print("🤖 Initializing RAG Chatbot (Stable Architecture)...")
    
    # 2. Load Documents
    print(f"📂 Loading documents from '{DOCS_FOLDER}'...")
    try:
        loader = DirectoryLoader(DOCS_FOLDER, loader_cls=UnstructuredFileLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s).")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    # 3. Split Text
    print("✂️  Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # 4. Embeddings & Vector Store
    print("Creating vector index...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 5. Setup the LLM
    llm = ChatOllama(model=MODEL_NAME, temperature=0.3)

    # 6. Create the Chain
    print("Building chain...")
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
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
            print(f"\nAI: {response['answer']}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()