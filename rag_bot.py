import os
import sys

# --- AI & LangChain Libraries ---
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
# UPDATED IMPORT BELOW:
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- CONFIGURATION ---
DOCS_FOLDER = "source_docs"  # Folder containing your PDFs/Docs
MODEL_NAME = "mistral"       # The Ollama model to use

def main():
    # 1. Validation: Ensure the folder exists and has files
    if not os.path.exists(DOCS_FOLDER) or not os.listdir(DOCS_FOLDER):
        print(f"Error: Folder '{DOCS_FOLDER}' is empty or missing.")
        print("Action: Create a folder named 'source_docs' and put your PDFs inside.")
        return

    print("🤖 Initializing Local RAG Chatbot...")
    
    # 2. Load Documents
    print(f"Loading documents from '{DOCS_FOLDER}'...")
    try:
        # DirectoryLoader scans the folder. We use Unstructured to handle mixed file types.
        loader = DirectoryLoader(DOCS_FOLDER, loader_cls=UnstructuredFileLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s).")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    # 3. Split Text (Chunking)
    print("✂️  Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"   -> Created {len(chunks)} chunks of text.")

    # 4. Create Vector Index (Embeddings)
    print("🧠 Creating vector index (loading embedding model)...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={'normalize_embeddings': True}
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        print("Index created successfully!")
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return

    # 5. Setup the LLM (Brain)
    llm = ChatOllama(model=MODEL_NAME, temperature=0.3, base_url="http://localhost:11434")

    # 6. Setup the Retrieval Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # 7. Start the Chat Loop
    print("\n" + "="*50)
    print("Chatbot Ready! Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            
            if not query.strip():
                continue

            print("🤖 Thinking...", end="\r")
            
            # Run the query
            response = qa_chain.invoke({"query": query})
            answer = response['result']
            source_docs = response['source_documents']

            # Print Answer
            print(f"\nAI: {answer}\n")
            
            # Print Sources
            print("[Sources Used:]")
            unique_sources = set()
            for doc in source_docs:
                src = doc.metadata.get('source', 'Unknown')
                if src not in unique_sources:
                    unique_sources.add(src)
                    print(f" - {src}")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()