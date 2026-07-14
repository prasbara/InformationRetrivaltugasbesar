import os
from typing import List
from langchain_core.documents import Document
from src.logger import log_system

PROMPT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "prompts", 
    "system_prompt.txt"
)

def load_system_prompt_template() -> str:
    """Loads the system prompt template from file."""
    if not os.path.exists(PROMPT_FILE):
        log_system(f"System prompt template file not found at {PROMPT_FILE}. Using fallback template.", "warning")
        return (
            "Anda adalah AI Campus Assistant. Jawablah pertanyaan HANYA menggunakan konteks di bawah.\n"
            "Jika jawaban tidak ada dalam konteks, jawab: 'Maaf, informasi tersebut tidak ditemukan pada dokumen yang tersedia sehingga saya tidak dapat memberikan jawaban.'\n"
            "Konteks:\n{context}\n\nPertanyaan: {question}\n\nSumber:"
        )
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        log_system(f"Error loading system prompt template: {str(e)}", "error")
        return ""

def format_context(chunks: List[Document]) -> str:
    """Formats retrieved chunks into a structured string for the LLM context."""
    formatted_chunks = []
    for chunk in chunks:
        meta = chunk.metadata
        source = meta.get("source", "Unknown Document")
        page = meta.get("page", "Unknown Page")
        chunk_idx = meta.get("chunk_index", "Unknown Chunk")
        
        chunk_header = f"[Dokumen: {source} | Halaman: {page} | Chunk: {chunk_idx}]"
        chunk_body = chunk.page_content
        
        formatted_chunks.append(f"{chunk_header}\n{chunk_body}\n")
        
    return "\n---\n".join(formatted_chunks)

def build_rag_prompt(chunks: List[Document], question: str) -> str:
    """Builds the final prompt string by merging formatted context and question into the system prompt template."""
    template = load_system_prompt_template()
    context_str = format_context(chunks)
    
    if not context_str:
        context_str = "[Tidak ada dokumen relevan yang ditemukan dalam database]"
        
    # Replace variables in template
    # Safe replacement in case there are other curly braces in the template
    prompt = template.replace("{context}", context_str).replace("{question}", question)
    return prompt
