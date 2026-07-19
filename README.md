# Local AI Campus Assistant 🎓

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: LangChain](https://img.shields.io/badge/Framework-LangChain-emerald?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langchain)
[![Database: ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-blueviolet)](https://github.com/chroma-core/chroma)
[![LLM Runtime: Ollama](https://img.shields.io/badge/Ollama-Llama_3-orange?logo=ollama&logoColor=white)](https://ollama.com/)
[![Open Source](https://img.shields.io/badge/Open_Source-❤-red)](https://github.com/)

A local Retrieval-Augmented Generation (RAG) system that provides accurate answers from university regulations and academic documents using semantic search and local LLM inference. Designed to run 100% privately on consumer hardware without leaking sensitive campus data.

---

## ✨ Features

- 🔋 **100% Local RAG Pipeline**: Complete privacy with local embeddings and LLM inference.
- 📂 **Multi-Format Knowledge Base**: Built-in support for parsing and clean indexing of `.pdf`, `.docx`, `.txt`, and `.md` files.
- 🔍 **Semantic Vector Search**: Dense retrieval powered by sentence-transformers and cosine similarity.
- 💾 **Persistent Vector Database**: Fast initialization and storage with a persistent ChromaDB instance.
- 🤖 **Ollama Integration**: Seamless integration with local runtimes to run state-of-the-art models like Llama 3.
- 🛡️ **Strict Prompt Guardrails**: Engineered prompt system that prevents hallucinations, rejects off-topic queries, and blocks prompt injection.
- 🏷️ **Citation Support**: Grounded AI responses featuring file-name, page, and chunk indexing citations.
- 🌊 **Streaming Responses**: Real-time word-by-word streaming generation for a smooth user experience.
- 🔄 **Automatic Document Indexing**: Streamlined pipelines to rebuild, clear, and analyze document collections via Web UI or CLI.
- 🧩 **Modular Architecture**: Clean, production-ready Python separation of concerns for loading, cleaning, chunking, and database operations.

---

## 🏗️ Architecture & Pipelines

This system decouples heavy offline pre-processing from fast online querying.

### 1. Offline Indexing Pipeline
Reads unstructured files, cleans them, chunks them to fit context windows, generates embeddings, and saves them to vector database.
```text
PDF / DOCX / TXT / MD
      ↓
[ Document Loader ] (Page & paragraph parsing)
      ↓
[ Text Cleaning ] (Whitespace normalization & noise filtering)
      ↓
[ RecursiveCharacterTextSplitter ] (Segmenting into overlapping chunks)
      ↓
[ Embedding Model ] (sentence-transformers/all-MiniLM-L6-v2)
      ↓
[ ChromaDB ] (Persistent vector database storage)
```

### 2. Online Retrieval Pipeline
Processes user queries in real-time, finds matching document chunks, constructs a strict prompt context, and obtains a structured response from the LLM.
```text
User Question
      ↓
[ Retriever (Cosine Similarity) ] (Calculates dense embeddings distances)
      ↓
[ Top-K Chunks ] (Retrieves best contextual fits)
      ↓
[ Prompt Builder ] (Assembles System Guardrails + Context + Query)
      ↓
[ LLM (Llama 3 via Ollama) ] (Fallback: OpenRouter API)
      ↓
[ Streaming Answer + Citation ] (Response printed dynamically with source metadata)
```

---

## 📦 Technology Stack

- **Language**: Python 3.12+
- **RAG Orchestrator**: LangChain Core & Community
- **Embeddings Generator**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector DB Store**: ChromaDB (Persistent DB)
- **Local Model Host**: Ollama (Llama 3 8B Instruct)
- **API Model Host (Fallback)**: OpenRouter (Llama 3 8B Instruct Cloud API)
- **Frontend Dashboard**: Streamlit (Premium UI layout)

---

## ⚙️ Configuration Parameters

### Embedding & Chunking Configuration
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model** | `sentence-transformers/all-MiniLM-L6-v2` | Light, fast, 384-dimensional dense retriever (~90MB) |
| **Chunk Size** | `700` | Max character length per text chunk |
| **Chunk Overlap** | `150` | Characters shared between neighboring chunks to preserve context |
| **Separators** | `["\n\n", "\n", " ", ""]` | Hierarchy used by recursive parser |

### LLM Inference Parameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Default Model** | `llama3` (or `meta-llama/llama-3-8b-instruct:free` via cloud) | Llama 3 8B Instruct |
| **Temperature** | `0.1` | Low creativity to enforce deterministic context grounding |
| **Top P** | `0.9` | Nucleus sampling probability threshold |
| **Max Tokens** | `1024` | Maximum length of generated answer |
| **Repeat Penalty** | `1.1` | Discourages repetitive text generations |
| **Top-K Retrieval** | `5` | Number of context chunks retrieved for query |
| **Context Window** | `4096` | Context window size in tokens (`num_ctx`) |

---

## 📂 Repository Structure

The repository is structured following academic guidelines alongside modern open-source conventions:

```text
InformationRetrivaltugasbesar/
│
├── docs/                      # Laporan & Panduan Pengguna (Academic Reports)
│   ├── LAPORAN TUBES INFORMATION RETRIVAL RAG.pdf   # Laporan tugas besar PDF
│   └── User_Manual.md         # Panduan cara instalasi & fitur aplikasi
│
├── code/                      # Aplikasi Utama RAG (System Source Code)
│   ├── app.py                 # Streamlit web interface & page routers
│   ├── config.yaml            # Config file mapping active parameters
│   ├── requirements.txt       # Python environment library packages
│   ├── .env.example           # Example config for API keys
│   │
│   ├── knowledge/             # Folder penyimpan dokumen akademik (PDF, DOCX, TXT, MD)
│   │   ├── Peraturan-Akademik-UNISSULA-2016.txt
│   │   └── PR-2-2021-Profil-CPL-MKWU-dan-Keg-Wajib-MHS.txt
│   │
│   ├── vector_db/             # Folder penyimpanan database ChromaDB (Generated)
│   │   └── chroma/
│   │
│   ├── src/                   # Python modular backend components
│   │   ├── chatbot.py         # Memory history buffers
│   │   ├── cleaner.py         # Text preprocessing and cleaning pipelines
│   │   ├── config.py          # Config file loaders
│   │   ├── embedding.py       # Embeddings weight loaders (offline-enforced)
│   │   ├── loader.py          # PDF/DOCX/TXT file parsers
│   │   ├── logger.py          # Multi-channel system logs
│   │   ├── prompt.py          # Prompt engineering & formatting
│   │   ├── rag.py             # LLM API connection adapters (Ollama & OpenRouter)
│   │   ├── retriever.py       # Cosine Similarity Search logic
│   │   ├── splitter.py        # Recursive text chunking mechanics
│   │   └── utils.py           # Shared utilities (Metric helpers & HTML charts)
│   │
│   ├── scripts/               # Automation scripts
│   │   └── reindex.py         # Terminal utility to index knowledge base
│   │
│   ├── models/                # Placeholder for local model downloads
│   ├── examples/              # Usage scripts and example notebooks
│   ├── tests/                 # QA Unit testing files
│   └── logs/                  # System, chat, and retrieval logs (Generated)
│
├── LICENSE                    # MIT Open-source License
└── README.md                  # Dokumentasi Utama Repositori
```

---

## 📈 RAG Knowledge & Control Flow

The operational data path for indexing documents and generating replies:

```mermaid
graph TD
    A[Load Raw Documents] --> B[Whitespace & Line Cleaning]
    B --> C[Recursive Chunking]
    C --> D[Generate MiniLM Embeddings]
    D --> E[Store in Persistent ChromaDB]
    
    F[User Query Input] --> G[Retrieve Top-K Cosine Chunks]
    E -.-> G
    G --> H[Construct Grounded Prompt]
    H --> I[LLM Inference (Ollama/OpenRouter)]
    I --> J[Streaming Response with Citations]
```

---

## 🛡️ Guardrails & Safety Constraints

This system is engineered for **zero-hallucination** campus assistance:
1. **Context Grounding**: The system prompt forces the LLM to reply *only* based on the provided context. If the database does not contain the answer, the LLM outputs a standard, polite refusal.
2. **Strict Citation Checks**: Responses are coupled with exact metadata tags (document name, page number, chunk ID).
3. **No Hallucinated Rules**: AI will never guess or fabricate administrative regulations.
4. **Prompt Injection Blocks**: System constraints override user instructions; any attempt to hijack the LLM to output non-academic topics is caught and neutralized.

---

## ⚡ Performance Characteristics

- **Zero-indexing inference**: High-speed lookups using indexing mappings in persistent database collections. Embedding calculations are only performed once per document edit.
- **Batched operations**: Text document uploads are pushed in batches of 100 to prevent CPU throttling.
- **Fast Startup scanning**: Uses lazy metadata reading (file size, page counts) on system load so the application UI starts in milliseconds, keeping the heavy embedding model loading deferred until indexing or querying.

---

## 🚀 Getting Started

Please refer to the detailed [docs/User_Manual.md](docs/User_Manual.md) for quick-start directions covering virtual environments, Ollama local model downloads, custom document indexing, and user interface maps.

---

## 🔮 Future Roadmap

*   **Hybrid Search**: Integrate BM25 Keyword Search alongside Cosine Vector Similarity.
*   **Re-ranking Step**: Implement a Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`) to boost precision.
*   **Evaluation Suites**: Standardize automated RAG testing using RAGAS metric frameworks.
*   **Incremental Indexing**: Re-index only modified files instead of full-database wipes.
*   **Multi-Document Synthesis**: Answer comparisons and syntheses across multiple guidelines.
*   **OCR Support**: Ingest scanned files natively.
*   **Table & Asset Extraction**: Capture tabular administrative regulations accurately.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
