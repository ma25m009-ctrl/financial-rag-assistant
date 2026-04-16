"""
Financial RAG Pipeline
Core retrieval-augmented generation logic using LangChain + FAISS
"""

import os
import re
# import fitz  # PyMuPDF
import pdfplumber
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


# ─── Financial keyword categories ─────────────────────────────────────────────
FINANCIAL_KEYWORDS = {
    "risk": [
        "risk", "uncertainty", "exposure", "volatility", "default", "credit risk",
        "market risk", "liquidity risk", "operational risk", "regulatory risk",
        "geopolitical", "macroeconomic", "recession", "downturn", "headwind"
    ],
    "revenue": [
        "revenue", "net revenue", "gross revenue", "income", "sales", "turnover",
        "top-line", "growth", "CAGR", "year-over-year", "YoY", "quarterly"
    ],
    "profit": [
        "profit", "net income", "EBITDA", "EBIT", "operating income", "margin",
        "earnings", "EPS", "return on equity", "ROE", "ROI", "profitability"
    ],
    "debt": [
        "debt", "liability", "leverage", "borrowing", "bond", "credit facility",
        "loan", "obligation", "interest expense", "debt-to-equity", "leverage ratio"
    ],
    "outlook": [
        "guidance", "forecast", "outlook", "projection", "target", "expectation",
        "pipeline", "backlog", "future", "strategy", "initiative", "plan"
    ],
}

ALL_FINANCIAL_KEYWORDS = [kw for kwlist in FINANCIAL_KEYWORDS.values() for kw in kwlist]


@dataclass
class RetrievedChunk:
    content: str
    page: int
    source: str
    score: float
    keywords_found: List[str] = field(default_factory=list)
    chunk_index: int = 0


@dataclass
class RAGResult:
    answer: str
    chunks: List[RetrievedChunk]
    query: str
    model_used: str
    tokens_used: int = 0


# ─── PDF Loader ────────────────────────────────────────────────────────────────
class FinancialPDFLoader:
    """Loads PDFs with page tracking and table-aware extraction (pdfplumber only)."""

    def load(self, file_path: str) -> List[Document]:
        docs = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""

                # Extract tables
                tables = page.extract_tables()
                table_text = self._tables_to_text(tables)

                full_text = text + ("\n\n[TABLE DATA]\n" + table_text if table_text else "")

                if full_text.strip():
                    docs.append(Document(
                        page_content=full_text.strip(),
                        metadata={
                            "page": page_num,
                            "source": os.path.basename(file_path)
                        }
                    ))

        return docs

        
    def _tables_to_text(self, tables: List) -> str:
        result = []
        for table in tables:
            if not table:
                continue
            rows = []
            for row in table:
                cleaned = [str(cell).strip() if cell else "" for cell in row]
                rows.append(" | ".join(cleaned))
            result.append("\n".join(rows))
        return "\n\n".join(result)


# ─── Text Chunker ──────────────────────────────────────────────────────────────
class FinancialChunker:
    """Splits documents into overlapping chunks optimized for financial text."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        return chunks


# ─── Vector Store ──────────────────────────────────────────────────────────────
class FinancialVectorStore:
    """FAISS-backed vector store with sentence-transformer embeddings."""

    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.store: Optional[FAISS] = None
        self.chunks: List[Document] = []

    def build(self, chunks: List[Document]) -> None:
        self.chunks = chunks
        self.store = FAISS.from_documents(chunks, self.embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        if not self.store:
            raise RuntimeError("Vector store not built. Call build() first.")
        results = self.store.similarity_search_with_score(query, k=top_k)
        return results

    def save(self, path: str) -> None:
        if self.store:
            self.store.save_local(path)

    def load(self, path: str) -> None:
        self.store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)


# ─── Keyword Highlighter ───────────────────────────────────────────────────────
def extract_keywords(text: str) -> List[str]:
    found = []
    text_lower = text.lower()
    for kw in ALL_FINANCIAL_KEYWORDS:
        if kw.lower() in text_lower:
            found.append(kw)
    return list(set(found))


def highlight_keywords(text: str, category_filter: Optional[str] = None) -> str:
    """Returns text with HTML <mark> tags around financial keywords."""
    keywords = []
    if category_filter and category_filter in FINANCIAL_KEYWORDS:
        keywords = FINANCIAL_KEYWORDS[category_filter]
    else:
        keywords = ALL_FINANCIAL_KEYWORDS

    # Sort by length descending to match longer phrases first
    keywords = sorted(keywords, key=len, reverse=True)
    for kw in keywords:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(
            lambda m: f'<mark class="kw-highlight">{m.group()}</mark>', text
        )
    return text


# ─── LLM Answerer ─────────────────────────────────────────────────────────────
class FinancialLLM:
    """Wraps HuggingFace Inference API (free tier) or OpenAI."""

    SYSTEM_PROMPT = """You are a senior financial analyst assistant with expertise in reading annual reports, earnings calls, and financial disclosures.

Your task is to answer questions accurately using ONLY the provided context from the document.

Rules:
1. Base your answer strictly on the provided context.
2. If the context doesn't contain enough information, say so clearly.
3. Cite specific numbers, figures, and quotes when available.
4. Use financial terminology appropriately.
5. Structure your answer clearly with key points.
6. Always mention which section/page the information comes from.
"""

    def __init__(self, provider: str = "huggingface", api_key: str = ""):
        self.provider = provider
        self.api_key = api_key

    def answer(self, query: str, chunks: List[RetrievedChunk]) -> Tuple[str, int]:
        context = self._build_context(chunks)
        prompt = self._build_prompt(query, context)

        if self.provider == "openai":
            return self._call_openai(prompt)
        else:
            return self._call_huggingface(prompt)

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[SOURCE {i} | Page {chunk.page} | {chunk.source}]\n{chunk.content}"
            )
        return "\n\n" + "─" * 60 + "\n\n".join(parts)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""{self.SYSTEM_PROMPT}

DOCUMENT CONTEXT:
{context}

QUESTION: {query}

ANSWER (cite page numbers and sources):"""

    def _call_openai(self, prompt: str) -> Tuple[str, int]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.2,
            )
            answer = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens
            return answer, tokens
        except Exception as e:
            return f"OpenAI Error: {str(e)}", 0

    def _call_huggingface(self, prompt: str) -> Tuple[str, int]:
        """Uses HuggingFace Inference API (free)."""
        try:
            import requests
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Using Mistral-7B via HF Inference API
            API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 600,
                    "temperature": 0.2,
                    "do_sample": True,
                    "return_full_text": False,
                },
            }
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            result = response.json()

            if isinstance(result, list) and result:
                answer = result[0].get("generated_text", "").strip()
                return answer, len(prompt.split())
            elif isinstance(result, dict) and "error" in result:
                return f"Model loading... Please retry in 20 seconds. ({result['error']})", 0
            else:
                return str(result), 0
        except Exception as e:
            return f"HuggingFace API Error: {str(e)}", 0


# ─── Main Pipeline ─────────────────────────────────────────────────────────────
class FinancialRAGPipeline:
    """End-to-end RAG pipeline orchestrator."""

    def __init__(self, llm_provider: str = "huggingface", api_key: str = ""):
        self.loader = FinancialPDFLoader()
        self.chunker = FinancialChunker()
        self.vector_store = FinancialVectorStore()
        self.llm = FinancialLLM(provider=llm_provider, api_key=api_key)
        self.is_ready = False
        self.doc_stats: Dict = {}

    def ingest(self, file_path: str, progress_callback=None) -> Dict:
        """Load, chunk, embed, and index a PDF document."""
        if progress_callback:
            progress_callback(0.1, "📄 Loading PDF...")

        docs = self.loader.load(file_path)
        total_pages = len(docs)
        total_chars = sum(len(d.page_content) for d in docs)

        if progress_callback:
            progress_callback(0.35, "✂️  Splitting into chunks...")

        chunks = self.chunker.split(docs)

        if progress_callback:
            progress_callback(0.55, "🧠 Generating embeddings (this takes ~30s)...")

        self.vector_store.build(chunks)
        self.is_ready = True

        self.doc_stats = {
            "pages": total_pages,
            "chunks": len(chunks),
            "chars": total_chars,
            "source": os.path.basename(file_path),
            "keywords_found": self._doc_keywords(docs),
        }

        if progress_callback:
            progress_callback(1.0, "✅ Ready!")

        return self.doc_stats

    def query(self, question: str, top_k: int = 5) -> RAGResult:
        """Retrieve relevant chunks and generate a grounded answer."""
        if not self.is_ready:
            raise RuntimeError("Pipeline not ready. Call ingest() first.")

        raw_results = self.vector_store.retrieve(question, top_k=top_k)

        chunks = []
        for doc, score in raw_results:
            chunk = RetrievedChunk(
                content=doc.page_content,
                page=doc.metadata.get("page", 0),
                source=doc.metadata.get("source", "unknown"),
                score=float(score),
                keywords_found=extract_keywords(doc.page_content),
                chunk_index=doc.metadata.get("chunk_index", 0),
            )
            chunks.append(chunk)

        answer, tokens = self.llm.answer(question, chunks)

        return RAGResult(
            answer=answer,
            chunks=chunks,
            query=question,
            model_used=f"{self.llm.provider}",
            tokens_used=tokens,
        )

    def summarize(self) -> RAGResult:
        """Generate a structured financial summary of the document."""
        summary_query = (
            "Provide a comprehensive financial summary covering: "
            "1) Company overview and business segments, "
            "2) Revenue and profit trends with specific numbers, "
            "3) Key risks and challenges, "
            "4) Strategic outlook and guidance, "
            "5) Notable highlights or red flags."
        )
        return self.query(summary_query, top_k=8)

    def _doc_keywords(self, docs: List[Document]) -> Dict[str, int]:
        counts = {}
        full_text = " ".join(d.page_content for d in docs).lower()
        for category, keywords in FINANCIAL_KEYWORDS.items():
            count = sum(full_text.count(kw.lower()) for kw in keywords)
            counts[category] = count
        return counts
