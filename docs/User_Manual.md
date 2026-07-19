# 🎓 Local AI Campus Assistant - User Manual

This manual provides instructions for setting up, running, configuring, and troubleshooting the **Local AI Campus Assistant**, a Retrieval-Augmented Generation (RAG) system developed for university document question answering.

---

## 👥 Contributors & Academic Identifiers
This project was developed by:
*   **Nabiel Ilyasa Pradana** (NIM: `32602300046`)
*   **M Alden Arraihan Ibrahim** (NIM: `32602300059`)
*   **Aisyha Nurrahmah Ar-rabbani** (NIM: `32602300078`)

**Program Studi Teknik Informatika**, Fakultas Teknologi Industri,  
**Universitas Islam Sultan Agung Semarang**, 2026.

---

## 1. System Requirements
To run this application locally with high performance, ensure your system meets the following specifications:
*   **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS (12.0+).
*   **Python Version**: Python 3.12 or higher.
*   **RAM**: Minimum 8 GB (16 GB recommended for local model execution).
*   **Disk Space**: Minimum 10 GB (to accommodate local LLMs, embeddings models, and vector stores).
*   **CPU/GPU**: Modern Multi-core CPU; CUDA-capable NVIDIA GPU (optional but recommended for faster response times).

---

## 2. Installation Guide

### Step 2.1: Clone the Repository
Clone this repository to your local machine and navigate into the `code/` folder:
```bash
git clone https://github.com/prasbara/InformationRetrivaltugasbesar.git
cd InformationRetrivaltugasbesar/code
```

### Step 2.2: Set Up Virtual Environment (Recommended)
Create a Python virtual environment to prevent package version conflicts:
```bash
# Create environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/macOS
source venv/bin/activate
```

### Step 2.3: Install Dependencies
Install all required libraries (LangChain, Streamlit, ChromaDB, PyPDF, etc.):
```bash
pip install -r requirements.txt
```

### Step 2.4: Install and Start Ollama (For Local LLM Inference)
1.  Download Ollama from the official site: [https://ollama.com](https://ollama.com)
2.  Follow the installer instructions for your OS.
3.  Ensure the Ollama application is running in the background.
4.  Download the default `llama3` instruction model from your terminal:
    ```bash
    ollama pull llama3
    ```

### Step 2.5: Configure OpenRouter (Optional Fallback)
If your local hardware is limited, you can use OpenRouter's cloud API.
1.  Open the `.env` file in the `code/` folder (or copy `.env.example` to `.env`):
    ```env
    OPENROUTER_API_KEY=your_openrouter_api_key_here
    ```
2.  Replace the placeholder with your actual OpenRouter key.
3.  In the Streamlit settings page, switch the **Model Provider** to `openrouter`.

---

## 3. Running the Application

### Step 3.1: Add Academic Documents
*   Place your campus academic guidelines, brochures, or PDF/docx documents into the `code/knowledge/` directory.
*   By default, two UNISSULA academic documents are included:
    *   `Peraturan-Akademik-UNISSULA-2016.txt`
    *   `PR-2-2021-Profil-CPL-MKWU-dan-Keg-Wajib-MHS.txt`

### Step 3.2: Perform Document Indexing
To perform indexing and save the vector embeddings into the persistent vector database:
*   **Via CLI (Fast)**:
    Run the indexing script from your terminal:
    ```bash
    python scripts/reindex.py
    ```
*   **Via Web Interface**:
    1. Launch the Streamlit application.
    2. Go to the **Settings** page ➔ select the **Indexing & Database** tab.
    3. Click **🚀 Mulai Indexing Dokumen**.

### Step 3.3: Launch the Web Portal
Start the Streamlit web server:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 4. UI Features & Navigation

### 📊 Dashboard
Displays overall stats of the indexed knowledge base:
*   Total loaded documents, pages, and vector database chunks.
*   Detailed chunking analytics (average chunk size, character count).
*   Visual distribution charts representing the size of character blocks.

### 📁 Document Explorer
Allows administrative previews of raw texts:
*   View status metadata (Size, total pages, load status).
*   Extract text contents page-by-page.
*   Verify whether PDFs are scanned/corrupted or readable.

### 🧩 Chunk Explorer
Lets developers audit how text is segmented:
*   Inspect individual chunks, index numbers, and file metadata.
*   Filter chunks by specific source documents or key terms.

### 🔍 Semantic Vector Search
Tests search performance without generating conversational answers:
*   Performs pure Cosine Similarity search.
*   Displays matching chunks with normalized scores (0.0 to 1.0).

### 💬 AI Chat
The primary user interaction portal:
*   Context-aware query input.
*   Streams LLM answers.
*   Provides structured citations (Source file name, page numbers, and chunk references) to verify correctness and combat hallucination.

### 🔄 Pipeline Monitor
Visualizes the RAG execution graph:
*   Flow diagram mapping offline indexing and online retrieval.
*   Detailed inspection of the last processed user query, retrieved context chunks, and exact raw prompt submitted to the LLM.

### ⚙️ Settings
Enables real-time system configuration:
*   **RAG & LLM**: Adjust Temperature, Top P, Max Tokens, Repeat Penalty, and Context Window.
*   **Indexing**: Customize Chunk Size and Overlap, and execute database wipes.
*   **Model Provider**: Toggle between Ollama (local) and OpenRouter (cloud API).

---

## 5. Troubleshooting & FAQ

### Q1: The chatbot is generating answers outside the documents or making things up.
*   **Fix**: Ensure your context is loaded. In **Settings ➔ Parameter RAG & LLM**, set **Temperature** to `0.0` or `0.1` to force deterministic responses. The system prompts have strict guardrails telling the LLM to refuse queries if the context is empty.

### Q2: A PDF page is loaded but reports 0 pages and 0 characters.
*   **Fix**: This occurs when a PDF contains scanned images instead of text. PyPDF cannot read images. Convert the PDF to a plain text file `.txt` using an OCR tool or copy the contents manually, then place it inside the `code/knowledge/` folder and re-index.

### Q3: Ollama connection fails or gives connection timeouts.
*   **Fix**: Verify that Ollama is running (check system tray icon). If it is running, check if it's responding by opening `http://localhost:11434` in your browser. Ensure the model name in your config (`llama3`) matches the model installed. You can download other models using `ollama pull <model_name>`.

### Q4: I changed Chunk Size or Overlap but nothing changed.
*   **Fix**: Whenever you adjust chunking hyperparameters, you **must re-index** the database. Go to **Settings ➔ Indexing & Database**, click **🗑️ Hapus Database ChromaDB**, and then click **🚀 Mulai Indexing Dokumen** to rebuild the index.
