"""
Module: cleaner.py
Purpose: Cleans document text by normalizing whitespaces and filtering out noise to improve vector storage and similarity retrieval accuracy.
Inputs: List of LangChain Document objects or raw text strings.
Outputs: List of cleaned Document objects or cleaned text strings.
Workflow: Replaces non-breaking spaces, normalizes line endings, limits consecutive newlines, normalizes duplicate spaces, strips whitespace from each line, and filters out empty documents.
Dependencies: re, typing, langchain_core.documents, src.logger.
Complexity: Time: O(L) where L is the total length of text; Space: O(L) for creating the cleaned copy.
"""
import re
from typing import List
from langchain_core.documents import Document
from src.logger import log_system

def clean_text(text: str) -> str:
    """Cleans document text by normalizing whitespaces and filtering noise."""
    if not text:
        return ""
    
    # Replace non-breaking spaces with standard space
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    
    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Reduce consecutive newlines to maximum two to maintain paragraphs
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Reduce multiple spaces to single spaces
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Strip whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    
    # Double check for duplicate whitespaces
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def clean_documents(documents: List[Document]) -> List[Document]:
    """Cleans the text content of all documents in place and filters out empty documents."""
    cleaned_docs = []
    empty_count = 0
    
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        if cleaned_content:
            doc.page_content = cleaned_content
            cleaned_docs.append(doc)
        else:
            empty_count += 1
            
    if empty_count > 0:
        log_system(f"Filtered out {empty_count} empty pages/documents during cleaning.", "warning")
        
    return cleaned_docs
