import os
import shutil
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

from src.logger import log_system

COLLECTION_NAME = "campus_assistant"

def get_vector_store(embedding_model: Embeddings, persist_directory: str) -> Chroma:
    """Initializes and returns the Chroma vector database."""
    try:
        # Resolve absolute path relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_persist_path = os.path.join(project_root, persist_directory)
        os.makedirs(full_persist_path, exist_ok=True)
        
        log_system(f"Connecting to ChromaDB at: {full_persist_path}", "info")
        vector_store = Chroma(
            persist_directory=full_persist_path,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME
        )
        return vector_store
    except Exception as e:
        log_system(f"Failed to connect to ChromaDB: {str(e)}", "error")
        raise e

def is_db_empty(vector_store: Chroma) -> bool:
    """Checks if the vector database contains any documents."""
    try:
        data = vector_store.get()
        return len(data.get("ids", [])) == 0
    except Exception as e:
        log_system(f"Error checking if database is empty: {str(e)}", "warning")
        return True

BATCH_SIZE = 100  # Process chunks in batches to avoid memory spikes

def add_documents_to_db(vector_store: Chroma, chunks: List[Document]) -> bool:
    """Adds chunked documents to the Chroma vector database in batches for performance."""
    if not chunks:
        log_system("No chunks provided to add to ChromaDB.", "warning")
        return False
    try:
        total = len(chunks)
        log_system(f"Adding {total} chunks to ChromaDB in batches of {BATCH_SIZE}...", "info")
        for i in range(0, total, BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            vector_store.add_documents(batch)
            log_system(f"  Batch {i // BATCH_SIZE + 1}: added {len(batch)} chunks.", "info")
        log_system("Successfully added all chunks to database.", "info")
        return True
    except Exception as e:
        log_system(f"Error adding documents to ChromaDB: {str(e)}", "error")
        return False

def clear_db(persist_directory: str) -> bool:
    """Clears the vector database by deleting its persistent directory contents."""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_persist_path = os.path.join(project_root, persist_directory)
        
        log_system(f"Clearing vector database at {full_persist_path}", "info")
        if os.path.exists(full_persist_path):
            shutil.rmtree(full_persist_path)
            os.makedirs(full_persist_path, exist_ok=True)
            log_system("Database cleared and re-created successfully.", "info")
            return True
        return False
    except Exception as e:
        log_system(f"Error clearing vector database: {str(e)}", "error")
        return False

def get_all_chunks(vector_store: Chroma) -> List[Document]:
    """Retrieves all documents/chunks currently stored in the database."""
    try:
        data = vector_store.get()
        documents = []
        ids = data.get("ids", [])
        metadatas = data.get("metadatas", [])
        documents_text = data.get("documents", [])
        
        for idx in range(len(ids)):
            meta = metadatas[idx] if metadatas else {}
            text = documents_text[idx] if documents_text else ""
            documents.append(Document(page_content=text, metadata=meta))
            
        return documents
    except Exception as e:
        log_system(f"Error fetching all chunks from database: {str(e)}", "error")
        return []
