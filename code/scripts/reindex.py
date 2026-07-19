"""
Module: reindex.py
Purpose: Implements the offline pipeline to read documents, clean, chunk, embed, and index them into persistent ChromaDB storage.
Inputs: User configuration details from config.yaml, text documents from knowledge/.
Outputs: Log status prints to terminal, persistent SQLite/Chroma collection directories.
Workflow: Reads config.yaml, deletes old database files, loads documents, cleans texts, segments them into chunks, loads HuggingFace embedding weights, connects to vector database, writes embeddings in batches.
Dependencies: sys, os, yaml, internal helper modules.
Complexity: Time: O(N * D) where N is number of chunks and D is embedding generation time; Space: O(N) database memory blocks.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.loader import load_all_documents
from src.cleaner import clean_documents
from src.splitter import split_documents
from src.embedding import get_embedding_model
from src.vectordb import get_vector_store, clear_db, add_documents_to_db

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    print("=" * 55)
    print("  CAMPUS AI ASSISTANT — RE-INDEXING PIPELINE")
    print("=" * 55)

    # Step 1: Hapus database lama
    print("\n[1/6] Membersihkan database lama...")
    clear_db(config["database_path"])
    print("      OK - Database dihapus.")

    # Step 2: Baca dokumen
    print("\n[2/6] Membaca dokumen dari knowledge/...")
    docs, status = load_all_documents("knowledge/")
    print(f"      Ditemukan {len(status)} file, {len(docs)} halaman total.")
    for s in status:
        fname = s["filename"]
        stat  = s["status"]
        pages = s["pages"]
        print(f"      - {fname} [{stat}] {pages} halaman")

    if not docs:
        print("\nERROR: Tidak ada dokumen yang bisa dibaca!")
        print("       Pastikan file PDF/DOCX/TXT ada di folder 'knowledge/'")
        sys.exit(1)

    # Step 3: Cleaning
    print("\n[3/6] Membersihkan teks...")
    cleaned = clean_documents(docs)
    print(f"      OK - {len(cleaned)} halaman setelah cleaning.")

    # Step 4: Chunking
    print("\n[4/6] Memotong teks menjadi chunks...")
    chunks = split_documents(cleaned, config["chunk_size"], config["chunk_overlap"])
    print(f"      OK - {len(chunks)} chunks dibuat (size={config['chunk_size']}, overlap={config['chunk_overlap']})")

    # Step 5: Load embedding model
    print(f"\n[5/6] Memuat embedding model '{config['embedding_model']}'...")
    print("      (Mungkin lama saat pertama kali — model ~90MB)")
    embeddings = get_embedding_model(config["embedding_model"])
    print("      OK - Model dimuat.")

    # Step 6: Koneksi + Indexing
    print("\n[6/6] Mengindeks chunks ke ChromaDB...")
    vector_db = get_vector_store(embeddings, config["database_path"])
    success = add_documents_to_db(vector_db, chunks)

    print("\n" + "=" * 55)
    if success:
        print(f"  SELESAI! {len(chunks)} chunks dari {len(status)} dokumen berhasil diindeks.")
        print(f"  Database: {config['database_path']}")
    else:
        print("  GAGAL! Cek log untuk detail error.")
    print("=" * 55)

if __name__ == "__main__":
    main()
