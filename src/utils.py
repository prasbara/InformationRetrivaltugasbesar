import math
from typing import List, Dict, Any
from langchain_core.documents import Document

def format_size(bytes_size: int) -> str:
    """Formats file sizes into human-readable strings."""
    if bytes_size == 0:
        return "0 B"
    sizes = ["B", "KB", "MB", "GB"]
    i = int(math.floor(math.log(bytes_size, 1024)))
    p = math.pow(1024, i)
    s = round(bytes_size / p, 2)
    return f"{s} {sizes[i]}"

def analyze_chunks(chunks: List[Document]) -> Dict[str, Any]:
    """Calculates statistics from document chunks."""
    if not chunks:
        return {
            "total_chunks": 0,
            "total_chars": 0,
            "avg_chunk_size": 0.0,
            "first_chunk": "",
            "last_chunk": "",
            "chunk_sizes": []
        }
        
    chunk_sizes = [len(c.page_content) for c in chunks]
    total_chars = sum(chunk_sizes)
    avg_chunk_size = total_chars / len(chunks)
    
    first_chunk = chunks[0].page_content if len(chunks) > 0 else ""
    last_chunk = chunks[-1].page_content if len(chunks) > 0 else ""
    
    return {
        "total_chunks": len(chunks),
        "total_chars": total_chars,
        "avg_chunk_size": round(avg_chunk_size, 2),
        "first_chunk": first_chunk,
        "last_chunk": last_chunk,
        "chunk_sizes": chunk_sizes
    }

def get_pipeline_html() -> str:
    """Returns a CSS styled HTML component to visualize the RAG pipeline."""
    return """
    <style>
        .pipeline-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            font-family: 'Inter', -apple-system, sans-serif;
            margin: 20px 0;
            padding: 20px;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            border: 1px solid #f0f0f0;
        }
        .pipeline-step {
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 90%;
            padding: 15px 20px;
            margin: 10px 0;
            background: #fdfdfd;
            border-radius: 8px;
            border-left: 5px solid #1E88E5;
            box-shadow: 0 2px 5px rgba(0,0,0,0.02);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .pipeline-step:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }
        .step-num {
            font-weight: bold;
            font-size: 1.1em;
            color: #1E88E5;
            background: #E3F2FD;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .step-content {
            flex: 1;
            margin-left: 20px;
        }
        .step-title {
            font-weight: 600;
            font-size: 1.05em;
            color: #2c3e50;
        }
        .step-desc {
            font-size: 0.85em;
            color: #7f8c8d;
            margin-top: 3px;
        }
        .arrow-down {
            color: #bdc3c7;
            font-size: 1.5em;
            margin: 5px 0;
        }
    </style>
    <div class="pipeline-container">
        <div class="pipeline-step" style="border-left-color: #3498db;">
            <div class="step-num">1</div>
            <div class="step-content">
                <div class="step-title">📂 Document Loading</div>
                <div class="step-desc">Membaca dokumen dari folder <code>documents/</code> (PDF, DOCX, TXT, MD).</div>
            </div>
        </div>
        <div class="arrow-down">⬇️</div>
        <div class="pipeline-step" style="border-left-color: #f1c40f;">
            <div class="step-num">2</div>
            <div class="step-content">
                <div class="step-title">🧼 Text Cleaning</div>
                <div class="step-desc">Normalisasi spasi, pembersihan whitespace duplikat, dan penyaringan noise teks.</div>
            </div>
        </div>
        <div class="arrow-down">⬇️</div>
        <div class="pipeline-step" style="border-left-color: #e67e22;">
            <div class="step-num">3</div>
            <div class="step-content">
                <div class="step-title">✂️ Text Chunking</div>
                <div class="step-desc">Pemisahan dokumen dengan <code>RecursiveCharacterTextSplitter</code> agar pas dalam context window.</div>
            </div>
        </div>
        <div class="arrow-down">⬇️</div>
        <div class="pipeline-step" style="border-left-color: #9b59b6;">
            <div class="step-num">4</div>
            <div class="step-content">
                <div class="step-title">🧠 Embedding Generation</div>
                <div class="step-desc">Mengubah teks chunk menjadi representasi vektor numerik dengan <code>all-MiniLM-L6-v2</code>.</div>
            </div>
        </div>
        <div class="arrow-down">⬇️</div>
        <div class="pipeline-step" style="border-left-color: #1abc9c;">
            <div class="step-num">5</div>
            <div class="step-content">
                <div class="step-title">💾 Vector Storage</div>
                <div class="step-desc">Menyimpan representasi vektor ke database persistent <code>ChromaDB</code> agar tidak perlu indeks ulang.</div>
            </div>
        </div>
        <div class="arrow-down">⬇️</div>
        <div class="pipeline-step" style="border-left-color: #2ecc71;">
            <div class="step-num">6</div>
            <div class="step-content">
                <div class="step-title">🔍 Semantic Retrieval</div>
                <div class="step-desc">Pencarian dokumen menggunakan Similarity Search berbasis jarak kosinus terhadap kueri pengguna.</div>
            </div>
        </div>
        <div class="arrow-down">⬇️</div>
        <div class="pipeline-step" style="border-left-color: #e74c3c;">
            <div class="step-num">7</div>
            <div class="step-content">
                <div class="step-title">🤖 LLM Guardrail & Generation</div>
                <div class="step-desc">Membangun prompt terstruktur dengan sitasi ketat dan mengirimkannya ke LLM (Ollama/OpenRouter).</div>
            </div>
        </div>
    </div>
    """
