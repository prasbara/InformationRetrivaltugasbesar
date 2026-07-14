import os
# Force HuggingFace to use local cache only — prevents hang on network check
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langchain_community.embeddings import HuggingFaceEmbeddings
from src.logger import log_system

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """Instantiates and returns the HuggingFaceEmbeddings model."""
    try:
        log_system(f"Loading embedding model: {model_name}", "info")
        # Load embedding model onto CPU/GPU. SentenceTransformers handles caching locally.
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}, # Use CPU by default for portability
            encode_kwargs={'normalize_embeddings': True} # Ensure cosine similarity
        )
        log_system(f"Embedding model loaded successfully: {model_name}", "info")
        return embeddings
    except Exception as e:
        log_system(f"Failed to load embedding model {model_name}: {str(e)}", "error")
        raise e
