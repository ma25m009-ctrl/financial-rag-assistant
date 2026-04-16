#  FinRAG : Financial Document Intelligence

> A production-grade Retrieval-Augmented Generation (RAG) system for financial document analysis.  
> Built with LangChain · FAISS · Sentence Transformers · Mistral-7B · Streamlit

---

##  Architecture Overview

```
PDF Upload
    │
    ▼
FinancialPDFLoader          ← PyMuPDF + pdfplumber (tables!)
    │
    ▼
FinancialChunker            ← RecursiveCharacterTextSplitter (800 chars, 150 overlap)
    │
    ▼
FinancialVectorStore        ← sentence-transformers/all-MiniLM-L6-v2 → FAISS index
    │
    ▼
Semantic Retrieval          ← Top-K similarity search (cosine distance)
    │
    ▼
FinancialLLM                ← Mistral-7B (HF Inference API) or GPT-3.5
    │
    ▼
Grounded Answer + Citations ← Page numbers, source chunks, keyword highlights
```

---

##  Quickstart

### 1. Clone & Setup

```bash
git clone <your-repo>
cd financial-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get a Free API Key

**HuggingFace (Free):**
1. Go to https://huggingface.co/settings/tokens
2. Create a token with `read` access
3. Paste it in the app sidebar

**OpenAI (Paid):**
- Get key from https://platform.openai.com/api-keys

### 3. Run

```bash
streamlit run app.py
```

Open: http://localhost:8501

---

##  Project Structure

```
financial-rag/
├── app.py                  # Streamlit UI (main entry point)
├── src/
│   └── rag_pipeline.py     # Core RAG pipeline
├── requirements.txt
└── README.md
```

---

##  Features

| Feature | Details |
|---|---|
| **PDF Loading** | PyMuPDF + pdfplumber with table extraction |
| **Chunking** | Recursive character splitting (800 chars, 150 overlap) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| **Vector DB** | FAISS (flat L2 index, cosine normalized) |
| **LLM** | Mistral-7B-Instruct via HuggingFace Inference API (free) |
| **Fallback LLM** | GPT-3.5-turbo via OpenAI |
| **Financial NLP** | 60+ keyword detection across 5 categories |
| **Citations** | Page numbers + chunk scores in every answer |
| **Summarization** | One-click structured financial summary |

---

##  Sample Questions to Try

- "What are the key risks mentioned in this report?"
- "Summarize the revenue trends with specific numbers"
- "What is the company's debt situation?"
- "What guidance or outlook does management provide?"
- "Are there any red flags or concerns raised?"
- "What business segments does the company operate in?"

---

##  Configuration

| Parameter | Default | Description |
|---|---|---|
| LLM Provider | `huggingface` | `huggingface` or `openai` |
| Top-K Chunks | 5 | Number of retrieved chunks per query |
| Chunk Size | 800 | Characters per chunk |
| Chunk Overlap | 150 | Overlap between consecutive chunks |
| Embedding Model | `all-MiniLM-L6-v2` | Fast, accurate 384-dim embeddings |

---

##  Tech Stack 

- **LangChain** — RAG orchestration, text splitting, document loading
- **FAISS** — Facebook AI Similarity Search (vector database)
- **sentence-transformers** — State-of-the-art text embeddings
- **Mistral-7B** — Open-source LLM via HuggingFace Inference API
- **Streamlit** — Production-ready Python web UI
- **PyMuPDF / pdfplumber** — PDF parsing with table support

---


### Multi-document support
```python
# Call ingest() multiple times — FAISS merges indexes
pipeline.ingest("report_2023.pdf")
pipeline.ingest("report_2024.pdf")
```

### Persist the vector index
```python
pipeline.vector_store.save("./faiss_index")
pipeline.vector_store.load("./faiss_index")
```

### Custom financial keywords
```python
# In rag_pipeline.py, add to FINANCIAL_KEYWORDS:
FINANCIAL_KEYWORDS["esg"] = ["carbon", "sustainability", "ESG", "net zero"]
```

---

This project directly demonstrates:
- ✅ **Generative AI** — LLM-powered financial Q&A
- ✅ **Information Retrieval** — Semantic search over financial documents
- ✅ **Information Extraction** — Keyword detection, citation extraction
- ✅ **Financial Domain** — Built for annual reports, earnings, 10-K filings
- ✅ **Production Engineering** — Modular pipeline, error handling, configurable parameters
