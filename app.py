"""
Financial RAG Assistant — Streamlit UI
Run: streamlit run app.py
"""

import os
import sys
import time
import tempfile
import streamlit as st
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from rag_pipeline import (
    FinancialRAGPipeline,
    highlight_keywords,
    FINANCIAL_KEYWORDS,
    RAGResult,
    RetrievedChunk,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinRAG — Financial Document Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #13161d;
    --surface2: #1a1e27;
    --border: #252a36;
    --accent: #4ade80;
    --accent2: #22d3ee;
    --accent3: #f59e0b;
    --text: #e8eaf0;
    --text-muted: #6b7280;
    --red: #f87171;
}

.stApp { background: var(--bg); }

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* Typography */
h1, h2, h3, h4 { font-family: 'DM Serif Display', serif !important; }
p, div, span, label { font-family: 'DM Sans', sans-serif !important; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }

/* Header */
.fin-header {
    background: linear-gradient(135deg, #13161d 0%, #0d1520 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.fin-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(74,222,128,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.fin-header h1 {
    font-size: 2.4rem;
    color: var(--text);
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.fin-header .subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin: 0;
}
.fin-badge {
    display: inline-block;
    background: rgba(74,222,128,0.12);
    color: var(--accent);
    border: 1px solid rgba(74,222,128,0.25);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 0.8rem;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.card-accent { border-left: 3px solid var(--accent); }
.card-warning { border-left: 3px solid var(--accent3); }
.card-info { border-left: 3px solid var(--accent2); }

/* Answer block */
.answer-block {
    background: linear-gradient(135deg, #131a1f 0%, #0f1520 100%);
    border: 1px solid rgba(34,211,238,0.2);
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    color: var(--text);
    font-size: 0.95rem;
    line-height: 1.75;
    white-space: pre-wrap;
}

/* Source chunks */
.chunk-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    font-size: 0.85rem;
    color: #b0b8c8;
    line-height: 1.65;
}
.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
}
.chunk-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent);
    background: rgba(74,222,128,0.08);
    padding: 2px 8px;
    border-radius: 4px;
}
.score-bar {
    height: 3px;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%);
    border-radius: 3px;
    margin-bottom: 0.6rem;
}

/* Keyword highlight */
mark.kw-highlight {
    background: rgba(245,158,11,0.2);
    color: #fbbf24;
    border-radius: 3px;
    padding: 0 2px;
}

/* Stats */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.stat-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.stat-value {
    font-size: 1.5rem;
    font-family: 'DM Serif Display', serif;
    color: var(--accent);
    display: block;
}
.stat-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* Keyword pills */
.kw-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 500;
    margin: 2px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-muted);
    border-radius: 7px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
}

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #0a0f0a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #22c55e !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(74,222,128,0.25) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* Progress */
.stProgress > div > div > div { background: var(--accent) !important; }

/* Suggested questions */
.sq-btn {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
    margin: 0.2rem;
    font-size: 0.8rem;
    color: var(--text-muted);
    cursor: pointer;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "pipeline": None,
        "doc_stats": None,
        "chat_history": [],
        "last_result": None,
        "highlight_mode": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="fin-badge">⚡ CONFIGURATION</div>', unsafe_allow_html=True)
    st.markdown("### 🔧 LLM Settings")

    llm_provider = st.selectbox(
        "LLM Provider",
        ["huggingface", "openai"],
        help="HuggingFace is free. OpenAI requires an API key."
    )

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="hf_... or sk-...",
        help="HuggingFace key from huggingface.co/settings/tokens"
    )

    st.markdown("---")
    st.markdown("### 🔍 Retrieval Settings")
    top_k = st.slider("Top-K Chunks", 3, 10, 5, help="Number of document chunks to retrieve")
    highlight = st.toggle("🔆 Keyword Highlighting", value=True)

    st.markdown("---")
    st.markdown("### 📊 Keyword Categories")
    kw_filter = st.selectbox(
        "Highlight Category",
        ["all"] + list(FINANCIAL_KEYWORDS.keys()),
        help="Filter which financial keywords to highlight"
    )

    if st.session_state.doc_stats:
        st.markdown("---")
        st.markdown("### 📄 Document Stats")
        stats = st.session_state.doc_stats
        st.markdown(f"**Source:** `{stats['source']}`")
        st.markdown(f"**Pages:** {stats['pages']}")
        st.markdown(f"**Chunks:** {stats['chunks']}")
        st.markdown(f"**Characters:** {stats['chars']:,}")

        st.markdown("**Keyword Density:**")
        kw_data = stats.get("keywords_found", {})
        total_kw = sum(kw_data.values()) or 1
        colors = {"risk": "#f87171", "revenue": "#4ade80", "profit": "#22d3ee",
                  "debt": "#f59e0b", "outlook": "#a78bfa"}
        for cat, count in sorted(kw_data.items(), key=lambda x: -x[1]):
            pct = count / total_kw * 100
            c = colors.get(cat, "#6b7280")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin:3px 0">'
                f'<span style="color:{c};font-size:0.8rem;text-transform:capitalize">{cat}</span>'
                f'<span style="font-family:monospace;font-size:0.75rem;color:#9ca3af">{count}</span></div>',
                unsafe_allow_html=True
            )
            st.progress(pct / 100)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fin-header">
    <div class="fin-badge">FIN-RAG PROJECT • RAG + LLM</div>
    <h1>📊 FinRAG</h1>
    <p class="subtitle">Financial Document Intelligence — Upload reports, ask questions, get cited answers.</p>
</div>
""", unsafe_allow_html=True)


# ─── Upload Section ────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload Financial Document (PDF)",
        type=["pdf"],
        help="Annual reports, earnings releases, 10-K, investor presentations, news PDFs"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Reset Pipeline", use_container_width=True):
        st.session_state.pipeline = None
        st.session_state.doc_stats = None
        st.session_state.chat_history = []
        st.session_state.last_result = None
        st.rerun()


# ─── Ingest Pipeline ───────────────────────────────────────────────────────────
if uploaded_file and st.session_state.pipeline is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    pipeline = FinancialRAGPipeline(llm_provider=llm_provider, api_key=api_key)

    with st.spinner(""):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(pct, msg):
            progress_bar.progress(pct)
            status_text.markdown(f"**{msg}**")

        try:
            stats = pipeline.ingest(tmp_path, progress_callback=update_progress)
            st.session_state.pipeline = pipeline
            st.session_state.doc_stats = stats
            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ Indexed **{stats['pages']} pages** → **{stats['chunks']} chunks** from `{stats['source']}`")
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Ingestion failed: {str(e)}")


# ─── Main Interface ────────────────────────────────────────────────────────────
if st.session_state.pipeline:
    pipeline: FinancialRAGPipeline = st.session_state.pipeline

    # Doc stats overview
    stats = st.session_state.doc_stats
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-item"><span class="stat-value">{stats['pages']}</span><span class="stat-label">Pages</span></div>
        <div class="stat-item"><span class="stat-value">{stats['chunks']}</span><span class="stat-label">Chunks</span></div>
        <div class="stat-item"><span class="stat-value">{stats['chars']//1000}K</span><span class="stat-label">Characters</span></div>
        <div class="stat-item"><span class="stat-value">{sum(stats.get('keywords_found',{}).values())}</span><span class="stat-label">Financial Keywords</span></div>
    </div>
    """, unsafe_allow_html=True)

    tab_qa, tab_summary, tab_history = st.tabs(["💬 Q&A", "📋 Auto Summary", "🕘 History"])

    # ── Q&A Tab ────────────────────────────────────────────────────────────────
    with tab_qa:
        st.markdown("#### Ask a Question")

        # Suggested questions
        suggested = [
            "What are the key risks mentioned?",
            "Summarize the revenue trends",
            "What is the debt situation?",
            "What is the company's strategic outlook?",
            "Are there any profitability concerns?",
        ]
        st.markdown("**Suggested:**")
        cols = st.columns(len(suggested))
        for i, sq in enumerate(suggested):
            if cols[i].button(sq, key=f"sq_{i}", use_container_width=True):
                st.session_state["prefill_q"] = sq

        prefill = st.session_state.get("prefill_q", "")
        question = st.text_area(
            "Your question",
            value=prefill,
            placeholder="e.g., What revenue growth is projected? Are there any debt concerns?",
            height=90,
            label_visibility="collapsed",
        )
        if prefill:
            st.session_state["prefill_q"] = ""

        ask_col, _ = st.columns([1, 3])
        with ask_col:
            ask_clicked = st.button("🔍 Ask FinRAG", use_container_width=True)

        if ask_clicked and question.strip():
            with st.spinner("🔍 Retrieving context and generating answer..."):
                try:
                    result: RAGResult = pipeline.query(question.strip(), top_k=top_k)
                    st.session_state.last_result = result
                    st.session_state.chat_history.append({
                        "question": question.strip(),
                        "answer": result.answer,
                        "chunks": result.chunks,
                        "model": result.model_used,
                    })
                except Exception as e:
                    st.error(f"Query failed: {str(e)}")

        if st.session_state.last_result:
            result = st.session_state.last_result
            st.markdown("---")

            # Answer
            st.markdown("#### 🤖 Answer")
            st.markdown(
                f'<div class="answer-block">{result.answer}</div>',
                unsafe_allow_html=True
            )
            st.caption(f"Model: `{result.model_used}` • Chunks retrieved: {len(result.chunks)}" +
                       (f" • Tokens: {result.tokens_used}" if result.tokens_used else ""))

            # Source chunks
            st.markdown("#### 📎 Source Citations")
            for i, chunk in enumerate(result.chunks, 1):
                relevance = max(0, min(1, 1 - chunk.score / 2))  # normalize
                content_display = highlight_keywords(chunk.content, None if kw_filter == "all" else kw_filter) if highlight else chunk.content

                st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-header">
                        <strong style="color:#e8eaf0">Source {i}</strong>
                        <span class="chunk-meta">📄 {chunk.source} | Page {chunk.page} | Score: {chunk.score:.3f}</span>
                    </div>
                    <div class="score-bar" style="width:{int(relevance*100)}%"></div>
                    {content_display[:500]}{'...' if len(content_display) > 500 else ''}
                </div>
                """, unsafe_allow_html=True)

                if chunk.keywords_found:
                    colors = {"risk": "#f87171", "revenue": "#4ade80", "profit": "#22d3ee",
                              "debt": "#f59e0b", "outlook": "#a78bfa"}
                    pills = ""
                    for kw in chunk.keywords_found[:8]:
                        for cat, kwlist in FINANCIAL_KEYWORDS.items():
                            if kw.lower() in [k.lower() for k in kwlist]:
                                c = colors.get(cat, "#9ca3af")
                                pills += f'<span class="kw-pill" style="background:rgba(255,255,255,0.04);border:1px solid {c}30;color:{c}">{kw}</span>'
                                break
                    if pills:
                        st.markdown(f"<div style='margin-top:4px'>{pills}</div>", unsafe_allow_html=True)

    # ── Summary Tab ────────────────────────────────────────────────────────────
    with tab_summary:
        st.markdown("#### 📋 Automated Financial Summary")
        st.markdown("Generates a structured summary of the entire document using top-K retrieval across multiple financial dimensions.")

        if st.button("🚀 Generate Full Summary", use_container_width=False):
            with st.spinner("📊 Summarizing document..."):
                try:
                    result = pipeline.summarize()
                    st.markdown(f'<div class="answer-block">{result.answer}</div>', unsafe_allow_html=True)
                    st.caption(f"Based on {len(result.chunks)} retrieved chunks")
                except Exception as e:
                    st.error(f"Summary failed: {str(e)}")

    # ── History Tab ────────────────────────────────────────────────────────────
    with tab_history:
        st.markdown("#### 🕘 Conversation History")
        if not st.session_state.chat_history:
            st.info("No questions asked yet.")
        else:
            for i, entry in enumerate(reversed(st.session_state.chat_history), 1):
                with st.expander(f"Q{len(st.session_state.chat_history)-i+1}: {entry['question'][:80]}..."):
                    st.markdown(f"**Question:** {entry['question']}")
                    st.markdown(f'<div class="answer-block" style="margin-top:0.5rem">{entry["answer"]}</div>', unsafe_allow_html=True)
                    st.caption(f"Model: `{entry['model']}` | {len(entry['chunks'])} sources")

else:
    # Landing state
    st.markdown("""
    <div class="card card-info" style="text-align:center;padding:3rem">
        <h2 style="color:#e8eaf0;margin:0 0 0.5rem">Upload a financial document to begin</h2>
        <p style="color:#6b7280;margin:0">Supports annual reports, 10-K filings, earnings releases, investor presentations</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature grid
    features = [
        ("🧠", "RAG Pipeline", "LangChain + FAISS vector retrieval with semantic search"),
        ("📄", "PDF Intelligence", "PyMuPDF + pdfplumber for tables, text, and structured data"),
        ("🔦", "Keyword Analysis", "Auto-detects 60+ financial terms across 5 categories"),
        ("💬", "Cited Answers", "Every answer cites page numbers and source chunks"),
        ("📋", "Auto Summarize", "One-click financial summary across the full document"),
        ("🤖", "Dual LLM", "Mistral-7B (free) or GPT-3.5 — your choice"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="card" style="text-align:center">
                <div style="font-size:1.8rem">{icon}</div>
                <h4 style="color:#e8eaf0;margin:0.5rem 0 0.3rem">{title}</h4>
                <p style="color:#6b7280;font-size:0.82rem;margin:0">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
