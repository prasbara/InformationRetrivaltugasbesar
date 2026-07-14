import unittest
from langchain_core.documents import Document

from src.config import DEFAULT_CONFIG
from src.cleaner import clean_text
from src.splitter import split_documents
from src.prompt import build_rag_prompt

class TestRAGPipeline(unittest.TestCase):
    
    def test_default_config(self):
        """Tests that the default configurations have the correct schema keys."""
        self.assertIn("embedding_model", DEFAULT_CONFIG)
        self.assertIn("llm_provider", DEFAULT_CONFIG)
        self.assertEqual(DEFAULT_CONFIG["llm_provider"], "ollama")
        
    def test_clean_text(self):
        """Tests whitespace normalization and noise removal in cleaner module."""
        dirty_text = "Hallo   dunia!\n\n\n\nIni   tes  pembersihan."
        cleaned = clean_text(dirty_text)
        self.assertEqual(cleaned, "Hallo dunia!\n\nIni tes pembersihan.")
        
    def test_split_documents(self):
        """Tests document chunking using split_documents."""
        doc = Document(page_content="A " * 500, metadata={"source": "test.txt"})
        chunks = split_documents([doc], chunk_size=200, chunk_overlap=50)
        self.assertTrue(len(chunks) > 1)
        self.assertEqual(chunks[0].metadata["chunk_index"], 1)
        self.assertEqual(chunks[1].metadata["chunk_index"], 2)
        
    def test_build_rag_prompt(self):
        """Tests prompt construction with retrieved context chunks."""
        chunks = [
            Document(page_content="Rektor UNISSULA adalah...", metadata={"source": "akademik.pdf", "page": 2, "chunk_index": 5})
        ]
        query = "Siapa rektor UNISSULA?"
        prompt = build_rag_prompt(chunks, query)
        
        self.assertIn("akademik.pdf", prompt)
        self.assertIn("Halaman: 2", prompt)
        self.assertIn("Chunk: 5", prompt)
        self.assertIn("Rektor UNISSULA adalah...", prompt)
        self.assertIn(query, prompt)

if __name__ == "__main__":
    unittest.main()
