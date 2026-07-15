from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.logger import log_system

def split_documents(
    documents: List[Document], 
    chunk_size: int = 700, 
    chunk_overlap: int = 150
) -> List[Document]:
    """Splits a list of Documents into chunks using RecursiveCharacterTextSplitter."""
    if not documents:
        log_system("No documents provided for chunking.", "warning")
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    try:
        chunks = splitter.split_documents(documents)
        
        # Enrich metadata with chunk indices
        for idx, chunk in enumerate(chunks):
            # Create a shallow copy of metadata and add index
            new_metadata = chunk.metadata.copy()
            new_metadata["chunk_index"] = idx + 1
            chunk.metadata = new_metadata
            
        log_system(f"Split {len(documents)} document pages into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap}).", "info")
        return chunks
        
    except Exception as e:
        log_system(f"Error during chunking: {str(e)}", "error")
        return []
