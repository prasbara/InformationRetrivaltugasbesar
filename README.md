# Local AI Campus Assistant 🎓

Aplikasi AI Campus Assistant yang berjalan **100% secara lokal** menggunakan arsitektur **Retrieval-Augmented Generation (RAG)**. Aplikasi ini dirancang untuk menjawab pertanyaan seputar panduan akademik, administrasi kampus, dan profil kemahasiswaan berdasarkan dokumen yang disediakan secara akurat tanpa halusinasi.

---

## 🏗️ Arsitektur RAG

Aplikasi ini menggunakan alur data (pipeline) RAG yang modular:

```text
Load Document ➔ Cleaning ➔ Chunking ➔ Embedding ➔ Vector Database ➔ Retriever ➔ Prompt Builder ➔ LLM ➔ Response
```

1. **Load Document**: Membaca dokumen dari folder `documents/` (mendukung format PDF, DOCX, TXT, MD).
2. **Cleaning**: Normalisasi spasi, merapikan baris baru, dan membuang karakter sampah/noise.
3. **Chunking**: Memotong dokumen dengan `RecursiveCharacterTextSplitter` ke dalam ukuran chunk yang sesuai dengan batasan model (default: 700 karakter, overlap: 150).
4. **Embedding**: Mengubah teks chunk menjadi representasi vektor numerik menggunakan `sentence-transformers/all-MiniLM-L6-v2`.
5. **Vector Database**: Menyimpan representasi vektor ke database persistent `ChromaDB` agar tidak perlu melakukan indexing ulang saat aplikasi dinyalakan kembali.
6. **Retriever**: Melakukan pencarian kemiripan kosinus (Cosine Similarity Search) untuk mencari chunk dokumen teratas (Top K) yang paling relevan dengan kueri pengguna.
7. **Prompt Builder**: Menyisipkan chunk yang relevan ke dalam template prompt sistem (System Prompt) yang ketat untuk mencegah halusinasi.
8. **LLM**: Mengirim prompt ke LLM (default: `llama3` via Ollama lokal atau API OpenRouter) untuk menghasilkan jawaban terstruktur dengan sitasi sumber.
9. **Response**: Menampilkan jawaban secara streaming ke pengguna beserta sitasi detail (Nama Dokumen, Halaman, dan Chunk).

---

## 🛠️ Persyaratan Sistem

- Python 3.12+
- RAM: Minimal 8 GB (Direkomendasikan 16 GB untuk menjalankan LLM lokal dengan lancar)
- Penyimpanan: Ruang kosong minimal 10 GB (untuk model LLM Ollama dan database)
- OS: Windows / Linux / macOS

---

## 🚀 Cara Instalasi & Menjalankan Aplikasi

Ikuti langkah-langkah berikut untuk menjalankan aplikasi:

### 1. Klon Repositori & Masuk ke Direktori
Pastikan Anda berada di direktori proyek ini:
```bash
cd "d:\Codingan\TubesInformation Retrival"
```

### 2. Instal Dependensi Python
Instal seluruh pustaka yang diperlukan:
```bash
pip install -r requirements.txt
```

### 3. Konfigurasi Ollama (Lokal)
1. Unduh dan instal Ollama dari situs resmi: [https://ollama.com](https://ollama.com)
2. Jalankan aplikasi Ollama di komputer Anda.
3. Unduh model LLM default (`llama3`) melalui terminal/command prompt:
   ```bash
   ollama pull llama3
   ```
   *(Catatan: Anda juga bisa mengunduh model lain seperti `mistral`, `gemma3`, `phi3`, atau `qwen2.5` dan mengubahnya di pengaturan aplikasi).*

### 4. Konfigurasi OpenRouter (Opsional - Fallback)
Aplikasi ini mendukung integrasi dengan **OpenRouter API** sebagai alternatif jika perangkat lokal Anda berat untuk menjalankan LLM lokal. 
- API Key telah diatur secara otomatis pada file `.env`.
- Anda dapat memilih provider **OpenRouter** pada tab **Model Provider** di halaman **Settings** di Streamlit.

### 5. Menambahkan Dokumen Kampus
- Letakkan dokumen akademik Anda (PDF, DOCX, TXT, MD) ke dalam folder `documents/`.
- Secara default, file `PR-2-2021-Profil-CPL-MKWU-dan-Keg-Wajib-MHS.pdf` dan `Peraturan-Akademik-UNISSULA-2016.pdf` sudah diletakkan di sana.

### 6. Menjalankan Aplikasi Streamlit
Jalankan server Streamlit:
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser Anda pada alamat: `http://localhost:8501`.

### 7. Melakukan Indexing Pertama Kali
1. Buka aplikasi di browser.
2. Pergi ke halaman **Settings** ➔ tab **Indexing & Database**.
3. Klik tombol **🚀 Mulai Indexing Dokumen**.
4. Tunggu hingga progress bar selesai. Status database di sidebar akan berubah menjadi **Chroma (Aktif)**.
5. Pergi ke halaman **AI Chat** untuk mulai bertanya!

---

## 📁 Struktur Proyek

```text
rag-campus-assistant/
│
├── app.py                  # Aplikasi utama Streamlit
├── requirements.txt        # Dependensi Python
├── README.md               # Dokumentasi proyek
├── .env                    # Variabel lingkungan (OpenRouter API Key)
├── .env.example            # Contoh variabel lingkungan
├── config.yaml             # Konfigurasi default (Model, Chunk, Parameter LLM)
│
├── documents/              # Folder penyimpanan dokumen akademik (PDF, DOCX, TXT, MD)
├── database/
│   └── chroma/             # Penyimpanan persistent database ChromaDB
│
├── logs/                   # Folder penyimpanan log (chat.log, retrieval.log, system.log)
├── assets/                 # Folder aset statis
├── prompts/
│   └── system_prompt.txt   # File prompt sistem dengan guardrail anti-halusinasi
│
├── src/                    # Kode sumber modular
│   ├── loader.py           # Pembuat & pembaca dokumen
│   ├── cleaner.py          # Pembersihan teks
│   ├── splitter.py         # Pemotong teks (Chunking)
│   ├── embedding.py        # Pembuat embedding HuggingFace
│   ├── vectordb.py         # Pengelola koneksi ChromaDB
│   ├── retriever.py        # Logika similarity search
│   ├── prompt.py           # Pembuat prompt RAG
│   ├── rag.py              # Orchestration & pemanggilan LLM (Ollama/OpenRouter)
│   ├── chatbot.py          # Pengelola memori chat (ConversationBufferMemory)
│   ├── config.py           # Pengelola file config.yaml
│   ├── logger.py           # Konfigurasi logging sistem
│   └── utils.py            # Fungsi utilitas (Visualisasi HTML, format ukuran, dsb.)
│
└── tests/                  # Pengujian unit
    └── test_rag.py         # Unit testing pipeline RAG
```

---

## 🔍 Troubleshooting Umum

### 1. Chatbot Menjawab di luar Dokumen / Berhalusinasi
- **Solusi**: System prompt sudah sangat ketat. Namun, jika LLM masih berhalusinasi, kecilkan nilai `temperature` menjadi `0.0` pada halaman **Settings** tab **Parameter RAG & LLM** agar model bersikap deterministik dan hanya merujuk pada teks konteks.

### 2. Dokumen Tidak Bisa Dibaca (Halaman 0 atau Karakter 0)
- **Solusi**: Jika dokumen berupa PDF hasil scan (gambar), `pypdf` tidak dapat mengekstrak teksnya. Sesuai instruksi, ubahlah file PDF tersebut menjadi file `.txt` menggunakan software OCR atau salin teksnya secara manual ke format teks biasa, lalu letakkan di folder `documents/`.

### 3. Error Koneksi Ollama
- **Solusi**: Pastikan aplikasi Ollama sudah terbuka di background komputer Anda. Jika Anda mengaksesnya dari terminal lain atau VM, pastikan environment variable `OLLAMA_HOST=0.0.0.0` telah diset sebelum menjalankan Ollama.
- **Alternatif**: Ganti provider LLM ke **OpenRouter** pada tab **Model Provider** di halaman **Settings**.

### 4. Perubahan Chunk Size atau Overlap Tidak Berdampak
- **Solusi**: Setiap kali Anda mengubah parameter chunk size atau overlap, Anda **wajib** melakukan re-indexing ulang. Caranya: Pergi ke halaman **Settings** ➔ tab **Indexing & Database** ➔ klik **🗑️ Hapus Database ChromaDB** ➔ klik **🚀 Mulai Indexing Dokumen**.

### 5. Chat History Terlalu Lambat atau Habis Memori
- **Solusi**: Chat history menggunakan memori session-state Streamlit. Jika chat sudah terlalu panjang, klik tombol **Hapus Riwayat Chat 🔄** di bagian atas halaman **AI Chat** untuk menyegarkan memori.
