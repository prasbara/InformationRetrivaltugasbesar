from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from src.logger import log_retrieval, log_system

def retrieve_relevant_chunks(
    vector_store: Chroma, 
    query: str, 
    top_k: int = 5
) -> List[Document]:
    """Retrieves relevant chunks from ChromaDB for a query and normalizes the similarity score."""
    if not query.strip():
        return []
        
    try:
        # Chroma returns distance score (lower is closer / more similar)
        results = vector_store.similarity_search_with_score(query, k=top_k)
        
        retrieved_docs = []
        for doc, distance in results:
            # Normalize L2 distance to a 0-1 similarity score
            # A common formula for L2 normalization: similarity = 1 / (1 + distance)
            # If normalized embeddings are used, distance ranges from 0 to 2.
            # In that case: similarity = 1.0 - (distance / 2.0)
            similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
            
            # Enrich metadata with the score
            new_metadata = doc.metadata.copy()
            new_metadata["score"] = round(similarity, 4)
            
            new_doc = Document(page_content=doc.page_content, metadata=new_metadata)
            retrieved_docs.append(new_doc)
            
        # Log retrieval query and outcomes
        log_retrieval(query, top_k, retrieved_docs)
        return retrieved_docs
        
    except Exception as e:
        log_system(f"Error during retrieval: {str(e)}", "error")
        return []
