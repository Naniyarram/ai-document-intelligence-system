# ruff: noqa: E402
# ui/app.py — Streamlit frontend for the AI Document Intelligence Platform.

import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import tempfile
import html as _html
from pathlib import Path

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS — only what Streamlit's theme system cannot handle natively.
# Chat alignment, source citation tags, and quality badges have no native
# Streamlit equivalent. Everything else uses framework APIs or config.toml.
st.markdown("""
<style>
/* Chat message alignment — Streamlit has no native left/right bubble primitives. */
.msg-user { text-align: right; margin: 14px 0; }
.msg-user-bubble {
    display: inline-block;
    background: #1A2340;
    border: 1px solid #2A3460;
    border-radius: 14px 14px 2px 14px;
    padding: 10px 16px;
    max-width: 76%;
    font-size: 14px;
    line-height: 1.6;
    color: #C8D0E8;
    text-align: left;
}
.msg-asst { margin: 14px 0; }
.msg-asst-bubble {
    display: inline-block;
    background: #111318;
    border: 1px solid #1E2130;
    border-left: 3px solid #4F6EF7;
    border-radius: 2px 14px 14px 14px;
    padding: 12px 16px;
    max-width: 90%;
    font-size: 14px;
    line-height: 1.7;
    color: #C8D0E8;
}

/* Source citation tags. */
.src-row { margin-top: 6px; }
.src-tag {
    display: inline-block;
    background: #131722;
    border: 1px solid #22273D;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    color: #7C8FAE;
    font-family: monospace;
    margin: 2px 4px 2px 0;
}

/* Answer quality badges — colour-coded by score range. */
.badge {
    display: inline-block;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-family: monospace;
    margin: 2px 4px 2px 0;
    border: 1px solid;
}
.badge-good { color: #4ADE80; border-color: #1E3A2F; background: #0D1F18; }
.badge-mid  { color: #FBBF24; border-color: #3A2E1A; background: #1A1508; }
.badge-low  { color: #F87171; border-color: #3A1A1A; background: #1A0A0A; }
.badge-info { color: #7C8FAE; border-color: #22273D; background: #111318; }

/* Pipeline architecture stage rows. */
.stage-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 12px 0;
    border-bottom: 1px solid #1A1D27;
}
.stage-num  { font-size: 11px; font-weight: 700; color: #3D4459;
              font-family: monospace; min-width: 22px; padding-top: 2px; }
.stage-dot  { width: 8px; height: 8px; border-radius: 50%;
              margin-top: 4px; flex-shrink: 0; }
.stage-name { font-size: 14px; font-weight: 600; color: #E2E8F0; margin-bottom: 2px; }
.stage-stat { font-size: 12px; color: #7C8FAE; font-family: monospace; margin-bottom: 1px; }
.stage-detail { font-size: 11px; color: #3D4459; font-family: monospace; }

/* Hide Streamlit chrome — cleaner for portfolio. */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Constants ────────────────────────────────────────────────────────────────

SAMPLE_DOC_NAME = "sample_consulting_agreement.txt"
SAMPLE_DOC_PATH = Path(__file__).resolve().parent.parent / "data" / SAMPLE_DOC_NAME

# Three questions that demonstrate the RAG pipeline concretely on the sample.
# Chosen to show different retrieval challenges: numeric fact, multi-part,
# and a procedure question that requires understanding clause structure.
SAMPLE_QUESTIONS = [
    "What are the three payment milestones and their amounts?",
    "How can either party terminate this agreement, and what notice is required?",
    "What confidentiality obligations survive after the agreement ends?",
]


# ── Session State ─────────────────────────────────────────────────────────────

def _init():
    defaults = {
        "pipeline":             None,
        "pipeline_initialized": False,
        "chat_history":         [],
        "active_doc":           None,
        "indexed_files":        [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _pipeline():
    if not st.session_state.pipeline_initialized:
        with st.spinner("Initializing pipeline..."):
            try:
                from pipeline import DocumentPipeline
                st.session_state.pipeline = DocumentPipeline()
                st.session_state.pipeline_initialized = True
            except Exception as exc:
                st.error(f"Pipeline init failed: {exc}")
                return None
    return st.session_state.pipeline


# ── Sample Document ───────────────────────────────────────────────────────────

def _load_sample():
    """
    Index the bundled sample document if it isn't already in ChromaDB.

    This uses the real pipeline — same ingestion, chunking, embedding, and
    retrieval path as any user-uploaded document. It does NOT bypass or fake
    any stage.

    If the sample is already indexed (ChromaDB persists across restarts),
    the function skips re-indexing and just sets it as active.
    """
    pipe = _pipeline()
    if not pipe:
        return

    if not SAMPLE_DOC_PATH.exists():
        st.error(
            f"Sample document not found at `{SAMPLE_DOC_PATH}`. "
            "Check that `data/sample_consulting_agreement.txt` exists."
        )
        return

    # Check whether this document is already indexed in ChromaDB.
    # get_document_list() only returns documents indexed in the current
    # session. For persistence across restarts we check ChromaDB directly.
    existing_collections = pipe.embedder.get_collection_names()
    clean_name = pipe.embedder._clean_collection_name(SAMPLE_DOC_NAME)
    already_indexed = clean_name in existing_collections

    if already_indexed and SAMPLE_DOC_NAME not in pipe.indexed_documents:
        # Document is in ChromaDB but pipeline was just initialized.
        # Rebuild chunk cache so BM25 works (same path as _rebuild_chunks_from_chroma).
        rebuilt = pipe._rebuild_chunks_from_chroma(SAMPLE_DOC_NAME)
        if rebuilt:
            pipe.all_chunks[SAMPLE_DOC_NAME] = rebuilt
        # Reconstruct indexed_documents entry from ChromaDB collection size.
        pipe.indexed_documents[SAMPLE_DOC_NAME] = {
            "filename":            SAMPLE_DOC_NAME,
            "total_pages":         1,
            "visual_pages":        0,
            "table_pages":         0,
            "total_chunks":        len(rebuilt),
            "indexed_chunks":      len(rebuilt),
            "processing_time_sec": "—",
        }

    if not already_indexed:
        # First visit — index the document through the full pipeline.
        bar  = st.progress(0, text="Indexing sample document...")
        info = st.empty()

        def _progress(step, pct):
            bar.progress(pct / 100, text=step)
            info.caption(step)

        pipe.progress_callback = _progress
        try:
            pipe.index(
                file_path=str(SAMPLE_DOC_PATH),
                original_filename=SAMPLE_DOC_NAME,
            )
            bar.empty()
            info.empty()
        except Exception as exc:
            bar.empty()
            info.empty()
            st.error(f"Failed to index sample: {exc}")
            return

    st.session_state.active_doc   = SAMPLE_DOC_NAME
    st.session_state.chat_history = []
    pipe.reset_conversation()
    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    from config import Config

    with st.sidebar:
        st.markdown("### ◈ Document Intelligence")
        st.caption("Multimodal RAG · Hybrid Retrieval · Grounded Q&A")
        st.divider()

        # Sidebar document uploader and list
        pipe = _pipeline()

        # Upload
        st.markdown("**Upload a document**")
        uploaded = st.file_uploader(
            "file",
            type=Config.SUPPORTED_FORMATS,
            key="sidebar_uploader",
            label_visibility="collapsed",
            help=f"PDF, DOCX, XLSX, CSV, TXT, PNG, JPG — max {Config.MAX_UPLOAD_MB} MB",
        )
        if uploaded:
            size_mb = uploaded.size / 1_048_576
            st.caption(f"{uploaded.name} · {size_mb:.1f} MB")
            if size_mb > Config.MAX_UPLOAD_MB:
                st.error(f"Exceeds {Config.MAX_UPLOAD_MB} MB limit.")
            elif st.button("Index Document", key="btn_index_sidebar", type="primary", use_container_width=True):
                _do_index(uploaded)

        st.divider()

        # Document list
        st.markdown("**Indexed documents**")
        if pipe:
            docs = pipe.get_document_list()
            if not docs:
                st.caption("No documents indexed yet.")
            else:
                for name in docs:
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        is_active = st.session_state.active_doc == name
                        short = (name[:24] + "…") if len(name) > 26 else name
                        label = f"▸ {short}" if is_active else short
                        if st.button(label, key=f"sel_{name}", use_container_width=True):
                            st.session_state.active_doc   = name
                            st.session_state.chat_history = []
                            pipe.reset_conversation()
                            st.rerun()
                    with c2:
                        if st.button("✕", key=f"rm_{name}", help="Remove from index"):
                            pipe.remove_document(name)
                            if st.session_state.active_doc == name:
                                st.session_state.active_doc   = None
                                st.session_state.chat_history = []
                            st.rerun()

        st.divider()

        # Pipeline status — shows real runtime counts when a document is loaded.
        st.markdown("**Pipeline**")
        _render_pipeline_status()

        st.divider()

        with st.expander("Configuration", expanded=False):
            st.caption(f"**Backend:** `{Config.get_backend_name()}`")
            st.caption(f"**LLM:** `{Config.get_llm_model()}`")
            st.caption(f"**VLM:** `{Config.get_vlm_model()}`")
            st.caption(f"**Embedder:** `{Config.EMBEDDING_MODEL}`")
            st.caption(f"**Chunk size:** {Config.CHUNK_SIZE} tokens")
            st.caption(f"**Active doc:** `{st.session_state.active_doc or 'None'}`")


def _render_pipeline_status():
    """
    Show real runtime stats when a document is active.
    When idle, show grey indicators and status ('idle' / 'ready').
    """
    from config import Config
    pipe      = _pipeline()
    api_ready = bool(Config.get_api_key())
    active    = st.session_state.active_doc
    doc_info  = pipe.indexed_documents.get(active, {}) if (pipe and active) else {}
    last_ret  = _last_assistant_msg()
    ret_stats = last_ret.get("retrieval", {}) if last_ret else {}

    if not active or not doc_info:
        stages = [
            ("Ingestion", "idle"),
            ("VLM", "idle"),
            ("Chunker", "idle"),
            ("Embedder", "idle"),
            ("Retriever", "idle"),
            ("LLM", "ready" if api_ready else "no key"),
        ]
        for name, val in stages:
            st.caption(f"○ **{name}** `{val}`")
    else:
        pages     = doc_info.get("total_pages", 1)
        vlm_pages = doc_info.get("visual_pages", 0)
        chunks    = doc_info.get("total_chunks", 0)
        vecs      = doc_info.get("indexed_chunks", chunks)
        time_sec  = doc_info.get("processing_time_sec")
        time_str  = f" · {time_sec}s" if isinstance(time_sec, (int, float)) else ""
        llm_model = Config.get_llm_model().split("/")[-1][:24]
        ret_val   = f"top {ret_stats['total_candidates']} reranked" if ret_stats.get("total_candidates") else "top-k 5 · reranked"

        stages = [
            ("Ingestion", f"{pages} pages{time_str}"),
            ("VLM", f"{vlm_pages} scanned pages"),
            ("Chunker", f"{chunks} chunks"),
            ("Embedder", f"{vecs} vecs · 384d"),
            ("Retriever", ret_val),
            ("LLM", llm_model),
        ]
        for name, val in stages:
            st.caption(f"● **{name}** `{val}`")


def _do_index(uploaded):
    pipe = _pipeline()
    if not pipe:
        return

    from config import Config
    name   = uploaded.name
    suffix = Path(name).suffix

    if suffix.lower().lstrip(".") not in Config.SUPPORTED_FORMATS:
        st.error(f"Unsupported format: {suffix}")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    bar  = st.progress(0, text="Starting...")
    info = st.empty()

    def _progress(step, pct):
        bar.progress(pct / 100, text=step)
        info.caption(step)

    pipe.progress_callback = _progress

    try:
        result = pipe.index(file_path=tmp_path, original_filename=name)
        st.session_state.active_doc   = name
        st.session_state.chat_history = []
        pipe.reset_conversation()
        bar.empty()
        info.empty()
        st.success(
            f"**{name}** ready — "
            f"{result['total_pages']} pages · "
            f"{result['total_chunks']} chunks · "
            f"{result['processing_time_sec']}s"
        )
    except Exception as exc:
        bar.empty()
        info.empty()
        st.error(f"Indexing failed: {exc}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Main Content ──────────────────────────────────────────────────────────────

def render_main():
    if not st.session_state.active_doc:
        _welcome()
        return

    # Active document banner — minimal, not decorative.
    st.caption(f"Active document: `{st.session_state.active_doc}`")

    tab_qa, tab_extract, tab_anomaly, tab_pipeline, tab_eval = st.tabs([
        "Q&A", "Extract", "Anomaly", "Pipeline", "Evaluation",
    ])

    with tab_qa:
        render_qa()
    with tab_extract:
        render_extract()
    with tab_anomaly:
        render_anomaly()
    with tab_pipeline:
        render_pipeline()
    with tab_eval:
        render_evaluation()


def _welcome():
    """
    Welcome screen.
    Presents two immediate options: try the pre-indexed sample document,
    or upload a custom document directly.
    """
    from config import Config

    st.markdown("## AI Document Intelligence")
    st.markdown(
        "Multimodal RAG pipeline: multi-format ingestion → VLM layout OCR → "
        "semantic chunking → 384d vector embedding → BM25 + dense retrieval → "
        "cross-encoder reranking → grounded LLM generation with citations."
    )
    st.divider()

    col_main, col_sep, col_upload = st.columns([1, 0.05, 1])

    with col_main:
        st.markdown("#### Try sample document")
        st.caption(
            "Pre-built consulting agreement covering Scope, Compensation, "
            "Termination, Confidentiality, and Governing Law."
        )
        st.markdown("")
        if st.button(
            "→ Try Sample Document",
            type="primary",
            use_container_width=True,
        ):
            _load_sample()

    with col_sep:
        st.markdown(
            '<div style="height:170px; border-left:1px solid #1E2130; margin:0 auto;"></div>',
            unsafe_allow_html=True,
        )

    with col_upload:
        st.markdown("#### Upload custom document")
        st.caption(
            "PDF, DOCX, XLSX, CSV, TXT, PNG, JPG (up to 50 MB). "
            "Scanned pages and images processed by Qwen2.5-VL-7B."
        )
        uploaded_main = st.file_uploader(
            "Upload document",
            type=Config.SUPPORTED_FORMATS,
            key="welcome_uploader",
            label_visibility="collapsed",
            help=f"PDF, DOCX, XLSX, CSV, TXT, PNG, JPG — max {Config.MAX_UPLOAD_MB} MB",
        )
        if uploaded_main:
            size_mb = uploaded_main.size / 1_048_576
            st.caption(f"{uploaded_main.name} · {size_mb:.1f} MB")
            if size_mb > Config.MAX_UPLOAD_MB:
                st.error(f"Exceeds {Config.MAX_UPLOAD_MB} MB limit.")
            elif st.button("Index Document", key="btn_index_welcome", type="primary", use_container_width=True):
                _do_index(uploaded_main)

    st.divider()

    st.markdown("##### Pipeline stages")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Ingestion & VLM**")
        st.caption(
            "Parses multi-format files. "
            "Scanned pages and embedded images are routed to Qwen2.5-VL-7B "
            "for layout-aware transcription before chunking."
        )
    with c2:
        st.markdown("**Hybrid Retrieval**")
        st.caption(
            "Parallel BM25 sparse index and all-MiniLM-L6-v2 dense vector index. "
            "Candidates are merged, deduplicated, and scored by a "
            "ms-marco-MiniLM-L-6-v2 cross-encoder reranker."
        )
    with c3:
        st.markdown("**Grounded Generation**")
        st.caption(
            "Llama-3.3-70B / Gemma-4 generation strictly bound to retrieved context. "
            "Requires page and section citations with 10-turn conversation memory."
        )


# ── Q&A Tab ───────────────────────────────────────────────────────────────────

def render_qa():
    is_sample = st.session_state.active_doc == SAMPLE_DOC_NAME

    if not st.session_state.chat_history:
        if is_sample:
            # For the sample, show targeted questions that demonstrate real retrieval.
            st.markdown("**Example questions for this document:**")
            for i, q in enumerate(SAMPLE_QUESTIONS):
                if st.button(q, key=f"sample_q{i}", use_container_width=False):
                    _run(q, mode="qa")
                    st.rerun()
            st.caption(
                "These questions test multi-fact retrieval, clause lookup, "
                "and post-termination obligation extraction."
            )
        else:
            st.caption("Ask your first question about the document.")
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="msg-user">'
                    f'<div class="msg-user-bubble">{_html.escape(msg["content"])}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                body = _html.escape(msg["content"]).replace("\n", "<br>")
                st.markdown(
                    f'<div class="msg-asst">'
                    f'<div class="msg-asst-bubble">{body}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                _quality_strip(msg)
                _source_tags(msg.get("sources", []))

    st.divider()

    # Input row.
    c1, c2 = st.columns([6, 1])
    with c1:
        question = st.text_input(
            "question",
            placeholder="e.g. What are the payment terms?",
            label_visibility="collapsed",
            key="qa_input",
        )
    with c2:
        ask = st.button("Ask →", type="primary", use_container_width=True)

    # Generic quick questions — shown for any document.
    qc = st.columns(4)
    generic = [
        "Summarize this document",
        "What are the key entities?",
        "List all important dates",
        "What are the main topics?",
    ]
    clicked = None
    for i, (col, q) in enumerate(zip(qc, generic)):
        with col:
            if st.button(q, use_container_width=True, key=f"q{i}"):
                clicked = q

    to_ask = (question if ask and question else clicked)
    if to_ask:
        _run(to_ask, mode="qa")
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            p = _pipeline()
            if p:
                p.reset_conversation()
            st.rerun()


def _run(question, mode="qa"):
    pipe = _pipeline()
    if not pipe:
        return
    st.session_state.chat_history.append(
        {"role": "user", "content": question, "sources": []}
    )
    with st.spinner("Retrieving and generating..."):
        result = pipe.query(
            question=question,
            collection_name=st.session_state.active_doc,
            mode=mode,
        )
    st.session_state.chat_history.append({
        "role":      "assistant",
        "content":   result["answer"],
        "sources":   result.get("sources",   []),
        "model":     result.get("model"),
        "quality":   result.get("quality",   {}),
        "entities":  result.get("entities",  {}),
        "anomalies": result.get("anomalies", []),
        "retrieval": result.get("retrieval", {}),
    })


def _quality_strip(msg):
    q     = msg.get("quality") or {}
    conf  = q.get("confidence")
    faith = q.get("faithfulness")
    model = msg.get("model", "")

    def _cls(v):
        if v is None:
            return "badge-info"
        return "badge-good" if v >= 0.75 else ("badge-mid" if v >= 0.45 else "badge-low")

    parts = []
    if conf is not None:
        parts.append(f'<span class="badge {_cls(conf)}">conf {conf:.2f}</span>')
    if faith is not None:
        parts.append(f'<span class="badge {_cls(faith)}">faith {faith:.2f}</span>')
    if model:
        short = model.split("/")[-1][:30]
        parts.append(f'<span class="badge badge-info">{_html.escape(short)}</span>')
    if parts:
        st.markdown("".join(parts), unsafe_allow_html=True)


def _source_tags(sources):
    if not sources:
        return
    tags = ""
    for s in sources:
        label = f"{s['file']}  p.{s['page']}"
        if s.get("section"):
            label += f" · {s['section'][:28]}"
        tags += f'<span class="src-tag">{_html.escape(label)}</span>'
    st.markdown(
        f'<div class="src-row"><small style="color:#3D4459;">Sources</small> {tags}</div>',
        unsafe_allow_html=True,
    )


def _last_assistant_msg():
    return next(
        (m for m in reversed(st.session_state.chat_history) if m["role"] == "assistant"),
        None,
    )


# ── Extract Tab ───────────────────────────────────────────────────────────────

def render_extract():
    st.markdown("#### Structured Extraction")
    st.caption(
        "Pull entities, clauses, figures, or any structured information from the document."
    )

    templates = {
        "All entities (dates, names, amounts)": (
            "Extract all named entities: dates, organizations, people, money amounts, and key terms."
        ),
        "Key clauses (contracts)": (
            "List all key clauses with their clause numbers and brief descriptions."
        ),
        "Table of contents": "List all sections and headings found in this document.",
        "Important numbers & figures": (
            "Extract all numerical data, statistics, and financial figures."
        ),
        "Action items / deadlines": (
            "List all action items, deadlines, and obligations mentioned."
        ),
        "Custom": "",
    }

    choice = st.selectbox("Template", list(templates.keys()))
    if choice == "Custom":
        query = st.text_area(
            "What to extract:",
            placeholder="e.g. Extract all liability clauses...",
        )
    else:
        query = templates[choice]
        st.text_area("Query (editable)", value=query, key="ext_disp")

    if st.button("Extract", type="primary") and query:
        _run(query, mode="extract")
        st.rerun()

    last = _last_assistant_msg()
    if last:
        st.divider()
        st.markdown("**Extracted information**")
        st.markdown(last["content"])
        _quality_strip(last)
        _source_tags(last.get("sources", []))

        ents = last.get("entities") or {}
        if ents:
            st.divider()
            st.markdown("**Detected entities**")
            cols = st.columns(min(len(ents), 3))
            for i, (cat, items) in enumerate(ents.items()):
                with cols[i % 3]:
                    st.markdown(f"**{cat.replace('_', ' ').title()}**")
                    for item in items[:10]:
                        st.markdown(f"- {item}")


# ── Anomaly Tab ───────────────────────────────────────────────────────────────

def render_anomaly():
    st.markdown("#### Anomaly Detection")
    st.caption(
        "Flags outliers, duplicate values, missing fields, and inconsistent patterns. "
        "Best suited for invoices, financial statements, and tabular data."
    )

    if st.button("Scan for anomalies", type="primary"):
        _run(
            "Analyze all numerical data, dates, and entries in this document. "
            "Flag anything unusual: duplicate values, outlier amounts, missing fields, "
            "or inconsistent patterns. Be specific about what is unusual and why.",
            mode="anomaly",
        )
        st.rerun()

    last = _last_assistant_msg()
    if last:
        st.divider()
        st.markdown("**Analysis result**")
        st.markdown(last["content"])
        _quality_strip(last)
        _source_tags(last.get("sources", []))

        anomalies = last.get("anomalies") or []
        if anomalies:
            st.divider()
            st.markdown("**Statistical anomalies**")
            for a in anomalies:
                if a["severity"] == "HIGH":
                    st.error(
                        f"[HIGH] Value `{a['raw']}` — "
                        f"{a['deviation']}× std dev from mean ({a['mean']})"
                    )
                else:
                    st.warning(
                        f"[MEDIUM] Value `{a['raw']}` — "
                        f"{a['deviation']}× std dev from mean ({a['mean']})"
                    )


# ── Pipeline Tab ──────────────────────────────────────────────────────────────

def render_pipeline():
    """
    Pipeline architecture view.

    Shows static component descriptions alongside real runtime stats
    from the currently active document. No values are hardcoded —
    every number comes from pipe.indexed_documents or the last retrieval result.
    """
    pipe     = _pipeline()
    active   = st.session_state.active_doc
    doc_info = pipe.indexed_documents.get(active, {}) if (pipe and active) else {}
    last     = _last_assistant_msg()
    ret      = last.get("retrieval", {}) if last else {}

    st.markdown("#### Pipeline Architecture")
    st.caption(
        "6-stage multimodal RAG pipeline. Each stage has a defined input/output contract "
        "and degrades gracefully if a component is unavailable."
    )

    # Per-stage: (num, name, model/detail, runtime_stat, color)
    stages = [
        ("01", "MultiModal Ingestion",
         "PDF · DOCX · XLSX · CSV · TXT · PNG · JPG "
         "— classifies each page: text / scanned / image / table",
         f"{doc_info['total_pages']} pages · "
         f"{doc_info.get('visual_pages', 0)} visual"
         if doc_info.get("total_pages") else None,
         "#4F6EF7"),
        ("02", "VLM Processing",
         "Qwen2.5-VL-7B via OpenRouter "
         "— extracts text from scans, charts, diagrams, embedded tables",
         None,
         "#8B5CF6"),
        ("03", "Semantic Chunking",
         "LangChain RecursiveCharacterTextSplitter "
         "— 400-token chunks, 60-token overlap, tables kept atomic",
         f"{doc_info['total_chunks']} chunks"
         if doc_info.get("total_chunks") else None,
         "#06B6D4"),
        ("04", "Embedding & Indexing",
         "all-MiniLM-L6-v2 → 384-dim vectors "
         "→ ChromaDB PersistentClient (local, survives restarts)",
         f"{doc_info['indexed_chunks']} vectors"
         if doc_info.get("indexed_chunks") else None,
         "#10B981"),
        ("05", "Hybrid Retrieval",
         "BM25 (keyword) + dense (semantic) "
         "→ merged → cross-encoder/ms-marco-MiniLM-L-6-v2 reranker",
         (f"top {ret['total_candidates']} reranked  "
          f"(BM25 {ret.get('bm25_count', 0)} + dense {ret.get('dense_count', 0)})")
         if ret.get("total_candidates") else None,
         "#F59E0B"),
        ("06", "LLM Generation",
         "Llama-3.3-70B-Instruct via OpenRouter "
         "— context-only system prompt, 10-turn conversation memory",
         None,
         "#EF4444"),
    ]

    for num, name, detail, runtime_stat, color in stages:
        stat_html = (
            f'<div class="stage-stat">{_html.escape(runtime_stat)}</div>'
            if runtime_stat else ""
        )
        st.markdown(
            f'<div class="stage-row">'
            f'<span class="stage-num">{num}</span>'
            f'<span class="stage-dot" style="background:{color};"></span>'
            f'<div>'
            f'<div class="stage-name">{name}</div>'
            f'{stat_html}'
            f'<div class="stage-detail">{detail}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Document statistics — only rendered when data is real.
    if doc_info:
        st.markdown("**Document statistics**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pages",        doc_info.get("total_pages",         "—"))
        c2.metric("Visual pages", doc_info.get("visual_pages",        "—"))
        c3.metric("Chunks",       doc_info.get("total_chunks",        "—"))
        c4.metric("Index time",
                  f"{doc_info['processing_time_sec']}s"
                  if isinstance(doc_info.get("processing_time_sec"), (int, float))
                  else doc_info.get("processing_time_sec", "—"))

    # Retrieval breakdown — only rendered when we have actual data from a query.
    if ret.get("total_candidates"):
        st.divider()
        st.markdown("**Last query — retrieval breakdown**")
        c1, c2, c3 = st.columns(3)
        c1.metric("BM25 candidates",  ret.get("bm25_count",       0))
        c2.metric("Dense candidates", ret.get("dense_count",       0))
        c3.metric("Final (reranked)", ret.get("total_candidates",  0))


# ── Evaluation Tab ────────────────────────────────────────────────────────────

def render_evaluation():
    import json
    import pandas as pd

    st.markdown("#### RAG Evaluation")
    st.caption(
        "Benchmark results from `artifacts/evaluation/rag_evaluation_results.json`. "
        "Run `python scripts/evaluate_rag.py` to regenerate."
    )

    artifact = (
        Path(__file__).resolve().parent.parent
        / "artifacts" / "evaluation" / "rag_evaluation_results.json"
    )

    if not artifact.exists():
        st.warning(
            "Evaluation artifact not found.\n\n"
            "```\npython scripts/evaluate_rag.py\n```"
        )
        return

    try:
        report = json.loads(artifact.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not load evaluation artifact: {exc}")
        return

    ts     = report.get("evaluation_timestamp", "—")
    model  = report.get("llm_model", report.get("model_backend", "—"))
    n_q    = report.get("num_questions",          0)
    n_ok   = report.get("llm_answers_succeeded",  0)
    n_fail = report.get("llm_answers_failed",     0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Questions evaluated", n_q)
    c2.metric("LLM answers OK",      n_ok)
    c3.metric("LLM answers failed",  n_fail)
    st.caption(f"Timestamp: `{ts}` · Model: `{model}`")
    st.divider()

    def _pct(v):
        return f"{v * 100:.1f}%" if v is not None else "N/A"

    def _flt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    ragas        = report.get("tier2_ragas", {})
    ragas_agg    = report.get("aggregate_metrics", {})
    ragas_status = ragas.get("status", "unknown")
    ragas_n      = ragas.get("num_evaluated", 0)

    st.markdown("**LLM-as-Judge metrics** (ragas 0.2.x)")
    st.caption(
        "Faithfulness: claims grounded in retrieved context? "
        "Answer Relevancy: answer addresses the question? "
        "Context Recall: retrieved context covers ground truth?"
    )

    if ragas_status == "completed" and ragas_n > 0:
        faith  = ragas.get("faithfulness")    or ragas_agg.get("faithfulness")
        ansrel = ragas.get("answer_relevancy") or ragas_agg.get("answer_relevancy")
        ctxrec = ragas.get("context_recall")   or ragas_agg.get("context_recall")
        c1, c2, c3 = st.columns(3)
        c1.metric("Faithfulness",     _pct(faith),  help=f"n={ragas_n}")
        c2.metric("Answer Relevancy", _pct(ansrel), help=f"n={ragas_n}")
        c3.metric("Context Recall",   _pct(ctxrec), help=f"n={ragas_n}")
        st.caption(f"Evaluated on {ragas_n} of {n_q} questions.")
    else:
        st.info(f"ragas status: `{ragas_status}`")

    st.divider()

    t1 = report.get("tier1_aggregate", ragas_agg)
    st.markdown("**Retrieval & factuality** (deterministic, no LLM required)")
    st.caption(
        "Retrieval Recall@K: top-K chunks contain expected source section? "
        "Key Fact Recall: required entities/numbers present in the answer?"
    )
    c1, c2 = st.columns(2)
    c1.metric(
        "Retrieval Recall@K",
        _pct(t1.get("retrieval_recall_at_k")),
        help="1 if top-K chunks contain the reference section keyword",
    )
    c2.metric(
        "Key Fact Recall",
        _pct(t1.get("key_fact_recall")),
        help="Fraction of key entities/numbers found in the answer",
    )

    st.divider()

    with st.expander("Diagnostic: Lexical Baseline (not ROUGE-1)", expanded=False):
        st.caption(
            "Token overlap metrics. High recall expected with verbose LLM answers. "
            "Diagnostic signals only — not primary quality indicators."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Lexical Precision",
                  _pct(t1.get("lexical_precision", t1.get("answer_precision"))))
        c2.metric("Lexical Recall",
                  _pct(t1.get("lexical_recall", t1.get("answer_recall"))))
        c3.metric("Lexical F1",
                  _pct(t1.get("lexical_f1", t1.get("answer_f1"))))
        rto = t1.get("retrieval_token_overlap", t1.get("context_relevance"))
        st.metric("Retrieval Token Overlap", _pct(rto), help="Diagnostic only")

    per_q = report.get("per_question_results", [])
    if per_q:
        st.divider()
        with st.expander(
            f"Per-query inspection ({len(per_q)} questions)", expanded=False
        ):
            st.caption(
                "Retrieved context, generated answers, and per-question scores. "
                "Demonstrates evaluation methodology."
            )
            rows = []
            for r in per_q:
                lf1 = r.get("lexical_f1") or r.get("f1") or {}
                rows.append({
                    "ID":       r["id"],
                    "Diff":     r.get("difficulty", "?"),
                    "KFR":      f"{r.get('key_fact_recall', 0):.2f}",
                    "RHit":     "Y" if r.get("retrieval_recall_hit") else "N",
                    "Faith":    _flt(r.get("faithfulness")),
                    "AnswRel":  _flt(r.get("answer_relevancy")),
                    "CtxRec":  _flt(r.get("context_recall")),
                    "LexF1":    f"{lf1.get('f1', 0):.2f}",
                    "Status":   "FAIL" if r.get("llm_error") else "OK",
                    "Question": r["question"][:55],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            for r in per_q:
                with st.expander(
                    f"{r['id']} — {r['question'][:60]}", expanded=False
                ):
                    qa_col, ans_col = st.columns(2)
                    with qa_col:
                        st.markdown("**Question**")
                        st.write(r["question"])
                        st.markdown("**Ground truth**")
                        st.write(r["ground_truth"])
                        st.markdown("**Reference section**")
                        st.code(r.get("reference_section", "—"))
                    with ans_col:
                        st.markdown("**Generated answer**")
                        if r.get("llm_error"):
                            st.error(f"LLM failed: {r['llm_error']}")
                        elif r.get("answer"):
                            st.write(r["answer"])
                        else:
                            st.warning("No answer generated.")

                    st.markdown("**Retrieved contexts (top 3)**")
                    for ci, ctx in enumerate(r.get("retrieved_contexts", [])[:3], 1):
                        st.text_area(
                            f"Chunk {ci}", ctx[:400], height=80,
                            disabled=True, key=f"ctx_{r['id']}_{ci}",
                        )

                    st.markdown("**Scores**")
                    lf1 = r.get("lexical_f1") or r.get("f1") or {}
                    mc  = st.columns(5)
                    mc[0].metric("Key Facts",    f"{r.get('key_fact_recall', 0):.2f}")
                    mc[1].metric("RHit",         "Y" if r.get("retrieval_recall_hit") else "N")
                    mc[2].metric("Faithfulness", _flt(r.get("faithfulness")))
                    mc[3].metric("Ans Rel",      _flt(r.get("answer_relevancy")))
                    mc[4].metric("Lex F1",       f"{lf1.get('f1', 0):.2f}")

    with st.expander("Metric methodology", expanded=False):
        methodology = report.get("metric_methodology", {})
        if methodology:
            for metric, desc in methodology.items():
                st.markdown(f"**`{metric}`** — {desc}")
        else:
            st.caption("No methodology documentation in this artifact.")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    _init()

    from config import Config
    if not Config.get_api_key() or Config.get_api_key() in (
        "hf_your_token_here", "sk-or-your-key-here", ""
    ):
        st.warning(
            "**No API key configured.** "
            "Add `HF_API_KEY` or `OPENROUTER_API_KEY` to your `.env` file "
            "to enable LLM generation and VLM processing."
        )

    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
