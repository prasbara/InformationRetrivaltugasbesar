import streamlit as st
import os
import yaml
import pandas as pd
from dotenv import load_dotenv, find_dotenv

# Load .env from the nearest parent directory
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)

# Import internal modules
from src.config import load_config, save_config, get_openrouter_api_key
from src.loader import load_all_documents
from src.cleaner import clean_documents
from src.splitter import split_documents
from src.embedding import get_embedding_model
from src.vectordb import get_vector_store, is_db_empty, add_documents_to_db, clear_db, get_all_chunks
from src.retriever import retrieve_relevant_chunks
from src.prompt import build_rag_prompt
from src.rag import check_ollama_status, get_installed_ollama_models, query_llm_stream
from src.chatbot import ChatbotManager
from src.utils import format_size, analyze_chunks, get_pipeline_html

# Page configurations
st.set_page_config(
    page_title="Campus AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS
st.markdown("""
    <style>
        /* General background and typography styling */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        .main {
            background-color: #fcfcfc;
        }
        
        /* Metric card styling */
        .metric-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.02);
            border: 1px solid #f0f0f0;
            text-align: center;
            margin-bottom: 15px;
        }
        
        .metric-val {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1e293b;
        }
        
        .metric-lbl {
            font-size: 0.9rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 5px;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #ffffff;
        }
        
        /* Chat bubble custom styling */
        .chat-bubble-user {
            background-color: #f1f5f9;
            padding: 15px;
            border-radius: 15px 15px 0px 15px;
            margin-bottom: 12px;
            color: #1e293b;
            text-align: right;
            display: inline-block;
            float: right;
            width: fit-content;
            max-width: 80%;
        }
        
        .chat-bubble-ai {
            background-color: #e2f1ff;
            padding: 15px;
            border-radius: 15px 15px 15px 0px;
            margin-bottom: 12px;
            color: #0f172a;
            display: inline-block;
            float: left;
            width: fit-content;
            max-width: 80%;
            border-left: 4px solid #0284c7;
        }
        
        .chat-container {
            width: 100%;
            overflow: auto;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- CACHE METHODS -----------------

@st.cache_resource
def get_cached_embeddings(model_name: str):
    """Caches the HuggingFace embedding model loading — heavy, only loaded once."""
    return get_embedding_model(model_name)

@st.cache_resource
def get_cached_vector_db(model_name: str, database_path: str):
    """Caches the ChromaDB connection (separate from embedding cache)."""
    embeddings = get_cached_embeddings(model_name)
    return get_vector_store(embeddings, database_path)

def clear_vector_db_cache():
    """Clears only the ChromaDB connection cache, preserving the loaded embedding model."""
    get_cached_vector_db.clear()

# ----------------- INITIALIZE STATE -----------------

if "config" not in st.session_state:
    st.session_state.config = load_config()

if "chatbot_manager" not in st.session_state:
    st.session_state.chatbot_manager = ChatbotManager()

if "last_retrievals" not in st.session_state:
    st.session_state.last_retrievals = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

# Load configurations locally to prevent scope issues
config = st.session_state.config

# Import pypdf and chromadb locally for fast scanning on startup
from pypdf import PdfReader
import chromadb

def check_db_status_fast(database_path: str) -> tuple:
    """Checks database status and returns (initialized, total_chunks) in milliseconds without loading embedding function."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(project_root, database_path)
    sqlite_file = os.path.join(full_path, "chroma.sqlite3")
    if not os.path.exists(sqlite_file):
        return False, 0
    try:
        client = chromadb.PersistentClient(path=full_path)
        # Check if collection exists
        collections = client.list_collections()
        has_col = any(c.name == "campus_assistant" for c in collections)
        if not has_col:
            return False, 0
        col = client.get_collection(name="campus_assistant")
        count = col.count()
        return count > 0, count
    except Exception:
        return False, 0

def scan_documents_fast(directory_path: str) -> list:
    """Quickly scans directory for documents and retrieves basic metadata (name, size, pages) without text extraction."""
    status_list = []
    if not os.path.exists(directory_path):
        return []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isdir(file_path):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".pdf", ".docx", ".txt", ".md"]:
            continue
        try:
            size = os.path.getsize(file_path)
            pages = 1
            if ext == ".pdf":
                reader = PdfReader(file_path)
                pages = len(reader.pages)
            status_list.append({
                "filename": filename,
                "file_path": file_path,
                "status": "Success",
                "size_bytes": size,
                "pages": pages,
                "char_count": 0, # Estimated/will be loaded during preview
                "error": None
            })
        except Exception as e:
            status_list.append({
                "filename": filename,
                "file_path": file_path,
                "status": "Failed",
                "size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                "pages": 0,
                "char_count": 0,
                "error": str(e)
            })
    return status_list

# Establish ChromaDB status without loading the heavy embedding model on startup
if "db_initialized" not in st.session_state or "total_chunks_count" not in st.session_state:
    db_init, chunks_count = check_db_status_fast(config["database_path"])
    st.session_state.db_initialized = db_init
    st.session_state.total_chunks_count = chunks_count

# Load document lists for global stats/explorer using fast metadata scan
if "documents_loaded" not in st.session_state or st.session_state.documents_loaded is False:
    status = scan_documents_fast("documents/")
    st.session_state.load_status = status
    st.session_state.documents_loaded = True


# Check Ollama and API Keys
ollama_online = check_ollama_status()
openrouter_key = get_openrouter_api_key()

# ----------------- SIDEBAR NAVIGATION -----------------

st.sidebar.markdown("<div style='text-align: center; padding: 10px 0;'><h2 style='color:#0f172a; margin-bottom: 0;'>🎓 Campus AI</h2><p style='color:#64748b; font-size:0.85rem; margin-top:0;'>Assistant Akademik Lokal</p></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "PILIH HALAMAN:",
    [
        "📊 Dashboard", 
        "📁 Document Explorer", 
        "🧩 Chunk Explorer", 
        "🔍 Vector Search", 
        "💬 AI Chat", 
        "🔄 Pipeline Monitor", 
        "⚙️ Settings"
    ]
)

st.sidebar.markdown("---")

# Service Status Panel in Sidebar
st.sidebar.markdown("### Status Layanan")
if ollama_online:
    st.sidebar.success("🟢 Ollama: Online")
else:
    st.sidebar.error("🔴 Ollama: Offline")

if openrouter_key:
    st.sidebar.success("🔑 OpenRouter Key: Tersedia")
else:
    st.sidebar.warning("⚠️ OpenRouter Key: Kosong")

# Active Config display
st.sidebar.markdown("### Konfigurasi Aktif")
st.sidebar.info(
    f"**Provider LLM:** {config['llm_provider'].upper()}\n\n"
    f"**Model LLM:** {config['llm_model'] if config['llm_provider'] == 'ollama' else config['openrouter_model']}\n\n"
    f"**Model Embedding:** {config['embedding_model'].split('/')[-1]}\n\n"
    f"**Database:** {'Chroma (Aktif)' if st.session_state.db_initialized else 'Chroma (Kosong)'}"
)

# ----------------- PAGE HANDLERS -----------------

if page == "📊 Dashboard":
    st.title("📊 Dashboard Analisis & Statistik")
    st.write("Overview data dokumen akademik kampus yang telah diindeks ke dalam sistem RAG.")
    
    # Major Metrics Card Row
    col1, col2, col3 = st.columns(3)
    
    total_docs = len(st.session_state.load_status) if st.session_state.load_status else 0
    total_pages = sum(d["pages"] for d in st.session_state.load_status) if st.session_state.load_status else 0
    total_chunks = st.session_state.total_chunks_count
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{total_docs}</div>
                <div class="metric-lbl">Total Dokumen</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{total_pages}</div>
                <div class="metric-lbl">Total Halaman</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{total_chunks}</div>
                <div class="metric-lbl">Total Chunk Database</div>
            </div>
        """, unsafe_allow_html=True)

    # Database and indexing status warnings
    if total_docs > 0 and not st.session_state.db_initialized:
        st.warning("⚠️ Dokumen terdeteksi di direktori 'documents/', tetapi belum dimasukkan ke database ChromaDB. Silakan pergi ke halaman **Settings** untuk memulai proses indexing.")
    elif total_docs == 0:
        st.info("ℹ️ Belum ada dokumen di direktori 'documents/'. Silakan tambahkan file PDF, DOCX, TXT, atau MD ke folder tersebut.")

    # Detailed statistics on chunks
    if st.session_state.db_initialized and total_chunks > 0:
        st.markdown("### Analisis Detail Chunking")
        
        # Lazy-load chunks for analysis
        if "db_chunks" not in st.session_state or not st.session_state.db_chunks:
            with st.spinner("Memuat data analitik chunk dari database..."):
                try:
                    vector_db = get_cached_vector_db(config["embedding_model"], config["database_path"])
                    st.session_state.db_chunks = get_all_chunks(vector_db)
                except Exception as e:
                    st.error(f"Gagal memuat data analitik: {e}")
                    st.session_state.db_chunks = []
                    
        if st.session_state.db_chunks:
            stats = analyze_chunks(st.session_state.db_chunks)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Karakter", f"{stats['total_chars']:,}")
            c2.metric("Rata-rata Ukuran Chunk", f"{stats['avg_chunk_size']} Karakter")
            c3.metric("Ukuran Maksimal Parameter", f"{config['chunk_size']} Karakter")
            
            # Display first and last chunk samples
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Contoh Chunk Pertama:**")
                st.info(f"\"{stats['first_chunk'][:250]}...\"" if stats['first_chunk'] else "N/A")
            with col_right:
                st.markdown("**Contoh Chunk Terakhir:**")
                st.info(f"\"{stats['last_chunk'][:250]}...\"" if stats['last_chunk'] else "N/A")
                
            # Chart visualization for size distribution
            st.markdown("#### Visualisasi Distribusi Ukuran Chunk (Karakter)")
            if stats["chunk_sizes"]:
                sizes_df = pd.DataFrame(stats["chunk_sizes"], columns=["Ukuran Chunk"])
                st.area_chart(sizes_df, color="#1e88e5")
        else:
            st.write("*(Gagal memuat detail analisis chunk)*")
    else:
        st.write("*(Indeks database kosong atau belum diinisialisasi. Analisis detail chunk tidak tersedia)*")


elif page == "📁 Document Explorer":
    st.title("📁 Document Explorer")
    st.write("Kelola dan telusuri dokumen kampus yang tersedia di direktori `documents/`.")
    
    if not st.session_state.load_status:
        st.info("Tidak ada file yang ditemukan dalam direktori `documents/`. Format yang didukung: PDF, DOCX, TXT, MD.")
    else:
        # Display loaded documents in a nice clean table
        df_docs = pd.DataFrame(st.session_state.load_status)
        
        # Select and format columns for display
        df_display = df_docs[["filename", "status", "size_bytes", "pages", "error"]].copy()
        df_display.columns = ["Nama Dokumen", "Status Load", "Ukuran File", "Total Halaman", "Detail Error"]
        df_display["Ukuran File"] = df_display["Ukuran File"].apply(format_size)
        
        st.dataframe(df_display, use_container_width=True)
        
        st.markdown("### Pratinjau Teks Dokumen")
        selected_file = st.selectbox("Pilih dokumen untuk dipratinjau:", [d["filename"] for d in st.session_state.load_status if d["status"] == "Success"])
        
        if selected_file:
            # Find document info
            file_info = next((d for d in st.session_state.load_status if d["filename"] == selected_file), None)
            
            if file_info:
                file_path = file_info["file_path"]
                
                # Lazy load full text on demand
                if "preview_docs_cache" not in st.session_state:
                    st.session_state.preview_docs_cache = {}
                    
                if selected_file not in st.session_state.preview_docs_cache:
                    with st.spinner("Mengekstrak teks dokumen untuk pratinjau..."):
                        from src.loader import load_single_document
                        doc_pages = load_single_document(file_path)
                        # Clean them
                        from src.cleaner import clean_documents
                        doc_pages = clean_documents(doc_pages)
                        st.session_state.preview_docs_cache[selected_file] = doc_pages
                else:
                    doc_pages = st.session_state.preview_docs_cache[selected_file]
                    
                if not doc_pages:
                    st.error("Teks dokumen kosong atau tidak bisa diekstrak. Jika file PDF adalah hasil scan (gambar), silakan konversi PDF tersebut menjadi file TXT terlebih dahulu.")
                else:
                    st.success(f"Teks berhasil diekstrak! Ditemukan {len(doc_pages)} halaman.")
                    
                    # If document has multiple pages, show page selection
                    if len(doc_pages) > 1:
                        page_num = st.slider("Pilih Halaman:", 1, len(doc_pages), 1)
                        selected_page = doc_pages[page_num - 1]
                    else:
                        selected_page = doc_pages[0]
                        page_num = 1
                        
                    # Render metadata and contents
                    st.markdown(f"**Metadata Halaman {page_num}:**")
                    st.json(selected_page.metadata)
                    
                    st.markdown("**Isi Konten Teks Halaman:**")
                    st.text_area(
                        label="Teks Halaman",
                        value=selected_page.page_content,
                        height=300,
                        disabled=True,
                        label_visibility="collapsed"
                    )


elif page == "🧩 Chunk Explorer":
    st.title("🧩 Chunk Explorer")
    st.write("Melihat, menelusuri, dan memfilter data chunk dokumen akademik yang tersimpan di dalam database.")
    
    if not st.session_state.db_initialized:
        st.warning("Database kosong. Indeks dokumen terlebih dahulu pada menu **Settings**.")
    else:
        # Lazy-load chunks for explorer
        if "db_chunks" not in st.session_state or not st.session_state.db_chunks:
            with st.spinner("Memuat data chunk dari database..."):
                try:
                    vector_db = get_cached_vector_db(config["embedding_model"], config["database_path"])
                    st.session_state.db_chunks = get_all_chunks(vector_db)
                except Exception as e:
                    st.error(f"Gagal memuat chunk: {e}")
                    st.session_state.db_chunks = []
                    
        if not st.session_state.db_chunks:
            st.warning("Tidak ada chunk yang berhasil dimuat.")
        else:
            # Filter controls
            c1, c2 = st.columns([1, 2])
            with c1:
                doc_list = ["Semua Dokumen"] + list(set(chunk.metadata.get("source", "Unknown") for chunk in st.session_state.db_chunks))
                selected_doc = st.selectbox("Filter berdasarkan Dokumen:", doc_list)
            with c2:
                search_query = st.text_input("Cari kata/frasa dalam chunk:", "")
                
            # Apply filters
            filtered_chunks = st.session_state.db_chunks
            
            if selected_doc != "Semua Dokumen":
                filtered_chunks = [c for c in filtered_chunks if c.metadata.get("source") == selected_doc]
                
            if search_query:
                filtered_chunks = [c for c in filtered_chunks if search_query.lower() in c.page_content.lower()]
                
            st.write(f"Menampilkan **{len(filtered_chunks)}** dari total **{len(st.session_state.db_chunks)}** chunk.")
            
            # Display the chunks
            for idx, chunk in enumerate(filtered_chunks[:30]):  # Limit to 30 chunks for UI speed
                meta = chunk.metadata
                with st.expander(f"Chunk {meta.get('chunk_index', idx+1)} | {meta.get('source', 'N/A')} | Halaman {meta.get('page', 1)}"):
                    st.json(meta)
                    st.text_area(
                        label=f"Isi Chunk {idx+1}",
                        value=chunk.page_content,
                        height=150,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                    
            if len(filtered_chunks) > 30:
                st.info("💡 Menampilkan 30 chunk pertama. Gunakan filter dokumen atau kotak pencarian untuk mempersempit hasil pencarian.")


elif page == "🔍 Vector Search":
    st.title("🔍 Semantic Vector Search")
    st.write("Uji performa pencarian dokumen menggunakan pencarian semantik (Similarity Search).")
    
    if not st.session_state.db_initialized:
        st.warning("Database kosong. Indeks dokumen terlebih dahulu pada menu **Settings**.")
    else:
        c1, c2 = st.columns([3, 1])
        with c1:
            query = st.text_input("Masukkan kueri pencarian:", placeholder="Contoh: Berapa IPK minimum untuk kelulusan?")
        with c2:
            top_k = st.slider("Jumlah Hasil (Top K):", 1, 10, config["top_k_retrieval"])
            
        if st.button("Lakukan Pencarian Semantik", type="primary") and query:
            with st.spinner("Menghubungkan ke database dan mencari chunk paling relevan..."):
                try:
                    vector_db = get_cached_vector_db(config["embedding_model"], config["database_path"])
                    results = retrieve_relevant_chunks(vector_db, query, top_k)
                except Exception as e:
                    st.error(f"Gagal melakukan pencarian: {e}")
                    results = []
                
                if not results:
                    st.warning("Tidak ditemukan kecocokan dokumen.")
                else:
                    st.success(f"Ditemukan {len(results)} chunk dokumen yang cocok!")
                    
                    for idx, doc in enumerate(results):
                        meta = doc.metadata
                        score = meta.get("score", 0.0)
                        
                        # Score bar color based on relevance
                        if score > 0.75:
                            score_color = "green"
                        elif score > 0.5:
                            score_color = "orange"
                        else:
                            score_color = "red"
                            
                        st.markdown(f"""
                            <div style="padding: 10px; background-color: #f8fafc; border-left: 5px solid #3b82f6; border-radius: 4px; margin-bottom: 10px;">
                                <span style="font-weight: 600; color: #1e293b;">Cocok #{idx+1}</span> | 
                                <span>Dokumen: <b>{meta.get('source', 'Unknown')}</b></span> | 
                                <span>Halaman: <b>{meta.get('page', 'N/A')}</b></span> | 
                                <span>Chunk: <b>{meta.get('chunk_index', 'N/A')}</b></span> | 
                                <span style="color: {score_color}; font-weight: 600;">Score: {score}</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.text_area(
                            label=f"Isi Chunk Pencarian #{idx+1}",
                            value=doc.page_content,
                            height=120,
                            disabled=True,
                            label_visibility="collapsed"
                        )


elif page == "💬 AI Chat":
    st.title("💬 Chat Akademik RAG")
    st.write("Tanyakan perihal peraturan akademik, administrasi kampus, atau kegiatan mahasiswa di sini.")
    
    # Warning if DB is not indexed
    if not st.session_state.db_initialized:
        st.warning("⚠️ Database ChromaDB kosong! Silakan pergi ke halaman **Settings** untuk mengindeks dokumen agar AI dapat menjawab berdasarkan dokumen akademik.")
        
    # Clear chat helper
    if st.button("Hapus Riwayat Chat 🔄", type="secondary"):
        st.session_state.chatbot_manager.clear_history()
        st.session_state.last_retrievals = []
        st.session_state.last_query = ""
        st.session_state.last_prompt = ""
        st.success("Riwayat chat dibersihkan.")
        st.rerun()

    # Display chat history
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    history = st.session_state.chatbot_manager.get_history()
    
    for msg in history:
        # Check message type based on LangChain Message class name
        role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)
            
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    if prompt_input := st.chat_input("Tanyakan perihal dokumen di sini..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt_input)
            
        # Execute RAG pipeline
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # Step 1: Retrieve context (or empty list if db not ready)
            retrieved_chunks = []
            if st.session_state.db_initialized:
                with st.spinner("Mencari konteks dokumen akademik..."):
                    try:
                        vector_db = get_cached_vector_db(config["embedding_model"], config["database_path"])
                        retrieved_chunks = retrieve_relevant_chunks(vector_db, prompt_input, config["top_k_retrieval"])
                    except Exception as e:
                        st.error(f"Gagal menghubungkan ke database: {e}")
                        retrieved_chunks = []
                
            # Store in session state for pipeline monitor
            st.session_state.last_query = prompt_input
            st.session_state.last_retrievals = retrieved_chunks
            
            # Step 2: Build prompt
            final_prompt = build_rag_prompt(retrieved_chunks, prompt_input)
            st.session_state.last_prompt = final_prompt
            
            # Step 3: Call LLM and Stream Response
            full_response = ""
            try:
                # Retrieve parameters
                params = {
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                    "max_tokens": config["max_tokens"],
                    "repeat_penalty": config["repeat_penalty"],
                    "context_window": config["context_window"],
                }
                
                # Check provider model selection
                if config["llm_provider"] == "ollama":
                    model = config["llm_model"]
                    api_key = ""
                else:
                    model = config["openrouter_model"]
                    api_key = openrouter_key
                    
                # Stream generator
                stream_generator = query_llm_stream(
                    provider=config["llm_provider"],
                    model=model,
                    prompt=final_prompt,
                    params=params,
                    api_key=api_key
                )
                
                # Stream implementation in Streamlit
                for chunk in stream_generator:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                    
                # Update placeholder with final text
                response_placeholder.markdown(full_response)
                
                # Save conversation in memory
                st.session_state.chatbot_manager.process_chat_turn(
                    user_message=prompt_input,
                    response_text=full_response,
                    provider=config["llm_provider"],
                    model=model
                )
                
            except Exception as e:
                err_msg = f"Terjadi kesalahan koneksi LLM: {str(e)}"
                response_placeholder.error(err_msg)
                
                # Fallback to display helpful information if Ollama is offline
                if "Ollama" in err_msg and not ollama_online:
                    st.info("💡 Tips: Layanan Ollama tidak terdeteksi berjalan pada komputer Anda. Pastikan Ollama telah dijalankan di background, atau Anda dapat mengganti provider ke OpenRouter di halaman **Settings**.")


elif page == "🔄 Pipeline Monitor":
    st.title("🔄 RAG Pipeline Monitor")
    st.write("Visualisasi step-by-step arsitektur Retrieval-Augmented Generation (RAG) dan pelacakan interaksi terakhir.")
    
    # Render flowchart
    st.markdown(get_pipeline_html(), unsafe_allow_html=True)
    
    st.markdown("### Pelacakan Pipeline Terakhir")
    if not st.session_state.last_query:
        st.info("Kueri belum terdeteksi. Silakan lakukan chat akademik atau pencarian semantik terlebih dahulu.")
    else:
        st.markdown(f"**1. Kueri Pengguna (Input):**")
        st.code(st.session_state.last_query)
        
        st.markdown(f"**2. Dokumen Terkait Hasil Retrieval (Similarity Search):**")
        if not st.session_state.last_retrievals:
            st.warning("Tidak ada dokumen yang di-retrieve (Kemungkinan database kosong atau kueri tidak cocok).")
        else:
            for idx, doc in enumerate(st.session_state.last_retrievals):
                meta = doc.metadata
                st.markdown(f"- **Match #{idx+1}:** {meta.get('source')} (Halaman {meta.get('page')}, Chunk {meta.get('chunk_index')}) | Score: `{meta.get('score')}`")
                with st.expander(f"Intip Isi Chunk #{idx+1}"):
                    st.write(doc.page_content)
                    
        st.markdown(f"**3. Prompt Terakhir yang Dikirim ke LLM (System Prompt + Context + Query):**")
        with st.expander("Tampilkan Prompt Lengkap"):
            st.text(st.session_state.last_prompt)


elif page == "⚙️ Settings":
    st.title("⚙️ Pengaturan & Manajemen Sistem")
    st.write("Konfigurasi parameter RAG, LLM, Chunking, dan manajemen Vector Database.")
    
    # Save settings helper
    def update_config(new_params: dict):
        current_config = st.session_state.config.copy()
        current_config.update(new_params)
        if save_config(current_config):
            st.session_state.config = current_config
            st.success("Konfigurasi berhasil disimpan dan diperbarui!")
            st.rerun()
        else:
            st.error("Gagal menyimpan konfigurasi.")

    tab1, tab2, tab3 = st.tabs(["🎛️ Parameter RAG & LLM", "📂 Indexing & Database", "🔬 Model Provider"])
    
    with tab1:
        st.markdown("### Parameter LLM")
        t_temp = st.slider("Temperature:", 0.0, 1.0, float(config["temperature"]), 0.05, help="Semakin tinggi nilai, semakin kreatif model tetapi berpotensi memicu halusinasi.")
        t_top_p = st.slider("Top P:", 0.0, 1.0, float(config["top_p"]), 0.05)
        t_max_tokens = st.number_input("Max Tokens:", 256, 4096, int(config["max_tokens"]), 128)
        
        # Options specific to Ollama
        st.markdown("### Parameter Lanjutan Ollama")
        t_repeat_penalty = st.slider("Repeat Penalty:", 0.5, 2.0, float(config.get("repeat_penalty", 1.1)), 0.05)
        t_context_window = st.number_input("Context Window size (num_ctx):", 2048, 16384, int(config.get("context_window", 4096)), 1024)
        
        if st.button("Simpan Parameter LLM", type="primary"):
            update_config({
                "temperature": t_temp,
                "top_p": t_top_p,
                "max_tokens": t_max_tokens,
                "repeat_penalty": t_repeat_penalty,
                "context_window": t_context_window
            })
            
    with tab2:
        st.markdown("### Konfigurasi Chunking")
        t_chunk_size = st.slider("Chunk Size (Karakter):", 200, 2000, int(config["chunk_size"]), 50)
        t_chunk_overlap = st.slider("Chunk Overlap (Karakter):", 0, 500, int(config["chunk_overlap"]), 10)
        t_top_k = st.slider("Top K Retrieval:", 1, 10, int(config["top_k_retrieval"]))
        
        if st.button("Simpan Parameter Chunking", type="primary"):
            update_config({
                "chunk_size": t_chunk_size,
                "chunk_overlap": t_chunk_overlap,
                "top_k_retrieval": t_top_k
            })
            
        st.markdown("---")
        st.markdown("### Manajemen Indeks Dokumen")
        st.write("Gunakan menu ini untuk menghapus basis data saat ini dan membangun ulang indeks dengan ukuran chunk baru.")
        
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            if st.button("🗑️ Hapus Database ChromaDB", type="secondary"):
                if clear_db(config["database_path"]):
                    clear_vector_db_cache()  # Only clear DB cache, not embedding model
                    st.session_state.db_initialized = False
                    st.session_state.total_chunks_count = 0
                    st.session_state.db_chunks = []
                    st.success("Database ChromaDB berhasil dihapus!")
                    st.rerun()
                else:
                    st.error("Gagal menghapus database.")
                    
        with col_db2:
            if st.button("🚀 Mulai Indexing Dokumen", type="primary"):
                # Run full indexing pipeline
                with st.status("Memproses Dokumen RAG...", expanded=True) as status_box:
                    try:
                        status_box.update(label="1. Membaca Dokumen Kampus...", state="running")
                        docs, load_status = load_all_documents("documents/")
                        
                        if not docs:
                            raise ValueError("Tidak ada dokumen yang bisa dibaca. Pastikan dokumen diletakkan di direktori 'documents/'.")
                            
                        status_box.update(label="2. Membersihkan Teks Dokumen...", state="running")
                        cleaned_docs = clean_documents(docs)
                        
                        status_box.update(label="3. Memotong Teks Menjadi Chunk...", state="running")
                        chunks = split_documents(cleaned_docs, config["chunk_size"], config["chunk_overlap"])
                        
                        status_box.update(label="4. Menghubungkan ke ChromaDB & Membuat Embeddings...", state="running")
                        # Wipe database folder before re-indexing
                        clear_db(config["database_path"])
                        # Only clear the vector DB connection cache — embedding model stays cached!
                        clear_vector_db_cache()
                        # Reuse already-loaded embedding model, just reconnect to fresh DB
                        embeddings = get_cached_embeddings(config["embedding_model"])
                        new_vector_db = get_vector_store(embeddings, config["database_path"])
                        
                        status_box.update(label="5. Mengindeks Chunk ke ChromaDB...", state="running")
                        success = add_documents_to_db(new_vector_db, chunks)
                        
                        if success:
                            st.session_state.db_initialized = True
                            st.session_state.db_chunks = chunks
                            st.session_state.loaded_docs = docs
                            st.session_state.load_status = load_status
                            status_box.update(label="Indexing Selesai dengan Sukses!", state="complete")
                            st.success(f"Berhasil mengindeks {len(chunks)} chunk dari {len(load_status)} dokumen!")
                            st.rerun()
                        else:
                            raise Exception("Gagal memasukkan chunk ke ChromaDB.")
                    except Exception as ex:
                        status_box.update(label="Proses Indexing Gagal!", state="error")
                        st.error(f"Error detail: {ex}")
                        
    with tab3:
        st.markdown("### Pilihan Provider LLM")
        t_provider = st.selectbox("Pilih Provider LLM:", ["ollama", "openrouter"], index=0 if config["llm_provider"] == "ollama" else 1)
        
        # Check available Ollama models if online
        ollama_models = []
        if ollama_online:
            ollama_models = get_installed_ollama_models()
            
        if t_provider == "ollama":
            if not ollama_online:
                st.warning("⚠️ Layanan Ollama terdeteksi offline. Pastikan Ollama berjalan di background sebelum memilih Ollama sebagai provider.")
                t_model = st.text_input("Nama Model Ollama (Ketik manual jika offline):", config["llm_model"])
            else:
                if ollama_models:
                    # Select from installed models, defaulting to config or first model
                    default_idx = 0
                    if config["llm_model"] in ollama_models:
                        default_idx = ollama_models.index(config["llm_model"])
                    t_model = st.selectbox("Pilih Model Ollama Terinstall:", ollama_models, index=default_idx)
                else:
                    st.warning("Tidak ditemukan model yang terinstall di Ollama. Silakan jalankan 'ollama pull llama3' di terminal Anda.")
                    t_model = st.text_input("Nama Model Ollama:", config["llm_model"])
        else:
            t_model = st.text_input("Model OpenRouter:", config["openrouter_model"], help="Contoh: meta-llama/llama-3-8b-instruct:free, google/gemma-2-9b-it:free, dll.")
            
        # Model embedding selection
        st.markdown("### Model Embedding HuggingFace")
        t_embedding = st.text_input("Model Embedding:", config["embedding_model"], help="Ubah jika ingin menggunakan model embedding lain. Catatan: Penggantian embedding model mengharuskan Anda melakukan indexing ulang database!")
        
        if st.button("Simpan Pilihan Model & Provider", type="primary"):
            update_data = {
                "llm_provider": t_provider,
                "embedding_model": t_embedding
            }
            if t_provider == "ollama":
                update_data["llm_model"] = t_model
            else:
                update_data["openrouter_model"] = t_model
                
            update_config(update_data)
