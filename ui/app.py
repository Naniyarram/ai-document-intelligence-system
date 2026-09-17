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

# Minimal CSS — only the things Streamlit's theme system can't handle natively.
# Kept small and purposeful. No custom font imports, no framework-within-a-framework.
st.markdown("""
<style>
/* Chat layout — Streamlit has no native chat alignment primitives. */
.msg-user {
    text-align: right;
    margin: 12px 0;
}
.msg-user-bubble {
    display: inline-block;
    background: #1A2340;
    border: 1px solid #2A3460;
    border-radius: 14px 14px 2px 14px;
    padding: 10px 16px;
    max-width: 78%;
    font-size: 14px;
    line-height: 1.6;
    color: #C8D0E8;
    text-align: left;
}
.msg-asst {
    margin: 12px 0;
}
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

/* Source tags — inline pill label for page citations. */
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

/* Quality badges — colour coded by score range. */
.badge { display:inline-block; border-radius:4px; padding:2px 8px; font-size:11px; font-family:monospace; margin:2px 4px 2px 0; border:1px solid; }
.badge-good { color:#4ADE80; border-color:#1E3A2F; background:#0D1F18; }
.badge-mid  { color:#FBBF24; border-color:#3A2E1A; background:#1A1508; }
.badge-low  { color:#F87171; border-color:#3A1A1A; background:#1A0A0A; }
.badge-neutral { color:#7C8FAE; border-color:#22273D; background:#111318; }

/* Pipeline stage connector line. */
.stage-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 12px 0;
    border-bottom: 1px solid #1A1D27;
}
.stage-index {
    font-size: 11px;
    font-weight: 700;
    color: #3D4459;
    font-family: monospace;
    min-width: 22px;
    padding-top: 2px;
}
.stage-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-top: 4px;
    flex-shrink: 0;
}
.stage-name {
    font-size: 14px;
    font-weight: 600;
    color: #E2E8F0;
    margin-bottom: 2px;
}
.stage-detail {
    font-size: 12px;
    color: #4B5468;
    font-family: monospace;
}

/* Hide Streamlit's hamburger menu and footer — cleaner for portfolio. */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# Session state

def _init():
    defaults = {
        "pipeline": None,
        "pipeline_initialized": False,
        "chat_history": [],
        "active_doc": None,
        "indexed_files": [],
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


# Sidebar

def render_sidebar():
    from config import Config

    with st.sidebar:
        st.markdown("### ◈ Document Intelligence")
        st.caption("Multimodal RAG · Hybrid Retrieval · Grounded Q&A")
        st.divider()

        # Upload
        st.markdown("**Upload a document**")
        uploaded = st.file_uploader(
            "file",
            type=Config.SUPPORTED_FORMATS,
            label_visibility="collapsed",
            help=f"PDF, DOCX, XLSX, CSV, TXT, PNG, JPG — max {Config.MAX_UPLOAD_MB} MB",
        )
        if uploaded:
            size_mb = uploaded.size / 1_048_576
            st.caption(f"{uploaded.name} · {size_mb:.1f} MB")
            if size_mb > Config.MAX_UPLOAD_MB:
                st.error(f"Exceeds {Config.MAX_UPLOAD_MB} MB limit.")
            elif st.button("Index Document", type="primary", use_container_width=True):
                _do_index(uploaded)

        st.divider()

        # Document list
        st.markdown("**Indexed documents**")
        pipe = _pipeline()
        if pipe:
            docs = pipe.get_document_list()
            if not docs:
                st.caption("No documents indexed yet.")
            else:
                for name in docs:
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        is_active = st.session_state.active_doc == name
                        label = f"▸ {name}" if is_active else name
                        if st.button(label, key=f"sel_{name}", use_container_width=True):
                            st.session_state.active_doc = name
                            st.session_state.chat_history = []
                            pipe.reset_conversation()
                            st.rerun()
                    with c2:
                        if st.button("✕", key=f"rm_{name}", help="Remove"):
                            pipe.remove_document(name)
                            if st.session_state.active_doc == name:
                                st.session_state.active_doc = None
                                st.session_state.chat_history = []
                            st.rerun()

        st.divider()

        # Pipeline status — simple and informative
        st.markdown("**Pipeline status**")
        api_ready = bool(Config.get_api_key())
        stages = [
            ("Ingestion",  True,       "PDF · DOCX · XLSX · IMG"),
            ("VLM",        api_ready,  "Qwen2.5-VL-7B"),
            ("Chunker",    True,       "LangChain Recursive"),
            ("Embedder",   True,       "all-MiniLM-L6-v2"),
            ("Retriever",  True,       "BM25 + Dense + Reranker"),
            ("LLM",        api_ready,  "Llama-3.3-70B"),
        ]
        for name, ok, detail in stages:
            icon = "🟢" if ok else "⚫"
            st.caption(f"{icon} **{name}** — {detail}")

        st.divider()

        with st.expander("Configuration"):
            st.caption(f"**Backend:** `{Config.get_backend_name()}`")
            st.caption(f"**LLM:** `{Config.get_llm_model()}`")
            st.caption(f"**VLM:** `{Config.get_vlm_model()}`")
            st.caption(f"**Embedder:** `{Config.EMBEDDING_MODEL}`")
            st.caption(f"**Chunk size:** {Config.CHUNK_SIZE} tokens")
            st.caption(f"**Active doc:** `{st.session_state.active_doc or 'None'}`")


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
        st.session_state.active_doc    = name
        st.session_state.chat_history  = []
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


# Main content area

def render_main():
    st.markdown("## AI Document Intelligence")
    st.caption("Upload a document, ask questions, extract entities, detect anomalies.")

    if not st.session_state.active_doc:
        _welcome()
        return

    # Active document indicator
    st.info(f"**Active document:** `{st.session_state.active_doc}`")

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
    st.divider()

    st.markdown("### Turn unstructured documents into structured intelligence.")
    st.markdown(
        "Upload a PDF, Word doc, Excel sheet, or scanned image in the sidebar. "
        "The pipeline will ingest, chunk, embed, and index it — then let you "
        "ask questions, extract structured data, and detect anomalies, "
        "all with grounded source citations."
    )

    st.markdown("")  # spacing

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Hybrid Q&A**")
        st.caption(
            "BM25 + dense vector search + cross-encoder reranker. "
            "Answers grounded in document text with page and section citations."
        )
    with c2:
        st.markdown("**Multimodal Ingestion**")
        st.caption(
            "Qwen2.5-VL-7B extracts text from scanned pages, charts, and diagrams. "
            "All content — visual and digital — is indexed together."
        )
    with c3:
        st.markdown("**Extraction & Anomalies**")
        st.caption(
            "Pull named entities, dates, and figures. "
            "Detect statistical outliers, duplicate values, and inconsistent entries."
        )

    st.divider()
    st.caption("← Use the sidebar to upload a file and click **Index Document** to begin.")


# Q&A tab

def render_qa():
    if not st.session_state.chat_history:
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

    st.caption("Quick questions:")
    qc = st.columns(4)
    quick = [
        "Summarize this document",
        "What are the key entities?",
        "List all important dates",
        "What are the main topics?",
    ]
    clicked = None
    for i, (col, q) in enumerate(zip(qc, quick)):
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
    st.session_state.chat_history.append({"role": "user", "content": question, "sources": []})
    with st.spinner("Retrieving context and generating answer..."):
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
    q = msg.get("quality") or {}
    conf  = q.get("confidence")
    faith = q.get("faithfulness")
    model = msg.get("model", "")

    def _cls(v):
        if v is None:
            return "badge-neutral"
        if v >= 0.75:
            return "badge-good"
        if v >= 0.45:
            return "badge-mid"
        return "badge-low"

    parts = []
    if conf is not None:
        parts.append(f'<span class="badge {_cls(conf)}">conf {conf:.2f}</span>')
    if faith is not None:
        parts.append(f'<span class="badge {_cls(faith)}">faith {faith:.2f}</span>')
    if model:
        short = model.split("/")[-1][:30]
        parts.append(f'<span class="badge badge-neutral">{_html.escape(short)}</span>')
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
    st.markdown(f"<small style='color:#3D4459;'>Sources</small> {tags}", unsafe_allow_html=True)


# Extract tab

def render_extract():
    st.markdown("#### Structured Extraction")
    st.caption("Pull specific entities, clauses, numbers, or any structured information from the document.")

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
        query = st.text_area("What to extract:", placeholder="e.g. Extract all liability clauses...")
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


# Anomaly tab

def render_anomaly():
    st.markdown("#### Anomaly Detection")
    st.caption("Best for invoices, financial statements, and tabular data. Flags outliers, duplicates, and missing fields.")

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
                        f"[HIGH] Value `{a['raw']}` — {a['deviation']}× std dev from mean ({a['mean']})"
                    )
                else:
                    st.warning(
                        f"[MEDIUM] Value `{a['raw']}` — {a['deviation']}× std dev from mean ({a['mean']})"
                    )


def _last_assistant_msg():
    return next(
        (m for m in reversed(st.session_state.chat_history) if m["role"] == "assistant"),
        None,
    )


# Pipeline tab

def render_pipeline():
    st.markdown("#### Pipeline Architecture")
    st.caption(
        "6-stage multimodal RAG pipeline. Each stage has a defined input/output contract "
        "and a fallback path so the system degrades gracefully if any component is unavailable."
    )

    stages = [
        ("01", "MultiModal Ingestion",
         "PDF · DOCX · XLSX · CSV · TXT · PNG · JPG — classifies each page as text / scanned / image / table",
         "#4F6EF7"),
        ("02", "VLM Processing",
         "Qwen2.5-VL-7B via OpenRouter — extracts text from scanned pages, charts, diagrams, embedded tables",
         "#8B5CF6"),
        ("03", "Semantic Chunking",
         "LangChain RecursiveCharacterTextSplitter — 400-token chunks, 60-token overlap, tables kept atomic",
         "#06B6D4"),
        ("04", "Embedding & Indexing",
         "all-MiniLM-L6-v2 → 384-dim vectors → ChromaDB PersistentClient (local, survives restarts)",
         "#10B981"),
        ("05", "Hybrid Retrieval",
         "BM25 (keyword) + dense (semantic) → merged → cross-encoder/ms-marco-MiniLM-L-6-v2 reranker",
         "#F59E0B"),
        ("06", "LLM Generation",
         "Llama-3.3-70B-Instruct via OpenRouter — strict context-only prompt, 10-turn conversation memory",
         "#EF4444"),
    ]

    for num, name, detail, color in stages:
        st.markdown(
            f'<div class="stage-row">'
            f'<span class="stage-index">{num}</span>'
            f'<span class="stage-dot" style="background:{color};"></span>'
            f'<div>'
            f'<div class="stage-name">{name}</div>'
            f'<div class="stage-detail">{detail}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    pipe = _pipeline()
    if pipe and st.session_state.active_doc:
        info = pipe.indexed_documents.get(st.session_state.active_doc)
        if info:
            st.markdown("**Document statistics**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pages",        info.get("total_pages",         0))
            c2.metric("Visual pages", info.get("visual_pages",        0))
            c3.metric("Chunks",       info.get("total_chunks",        0))
            c4.metric("Index time",   f"{info.get('processing_time_sec', 0)}s")

    last = _last_assistant_msg()
    if last and last.get("retrieval"):
        ret = last["retrieval"]
        st.divider()
        st.markdown("**Last query — retrieval breakdown**")
        c1, c2, c3 = st.columns(3)
        c1.metric("BM25 candidates",   ret.get("bm25_count",       0))
        c2.metric("Dense candidates",  ret.get("dense_count",       0))
        c3.metric("Final (reranked)",  ret.get("total_candidates",  0))


# Evaluation tab

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
    n_q    = report.get("num_questions", 0)
    n_ok   = report.get("llm_answers_succeeded", 0)
    n_fail = report.get("llm_answers_failed", 0)

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
        "Faithfulness: are answer claims grounded in retrieved context? "
        "Answer Relevancy: does the answer address the question? "
        "Context Recall: does retrieved context cover the ground truth?"
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
        "Retrieval Recall@K: did the top-K retrieved chunks contain the expected source section? "
        "Key Fact Recall: were required entities and numbers present in the generated answer?"
    )
    c1, c2 = st.columns(2)
    c1.metric("Retrieval Recall@K", _pct(t1.get("retrieval_recall_at_k")),
              help="Keyword heuristic — 1 if top-K chunks contain the reference section")
    c2.metric("Key Fact Recall",    _pct(t1.get("key_fact_recall")),
              help="Fraction of expected entities/numbers present in the answer")

    st.divider()

    with st.expander("Diagnostic: Lexical Baseline (not ROUGE-1)", expanded=False):
        st.caption(
            "Token overlap metrics. High recall is expected with verbose LLM answers. "
            "Treat as diagnostic signals, not primary quality indicators."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Lexical Precision", _pct(t1.get("lexical_precision", t1.get("answer_precision"))))
        c2.metric("Lexical Recall",    _pct(t1.get("lexical_recall",    t1.get("answer_recall"))))
        c3.metric("Lexical F1",        _pct(t1.get("lexical_f1",        t1.get("answer_f1"))))
        rto = t1.get("retrieval_token_overlap", t1.get("context_relevance"))
        st.metric("Retrieval Token Overlap", _pct(rto), help="Diagnostic only")

    per_q = report.get("per_question_results", [])
    if per_q:
        st.divider()
        with st.expander(f"Per-query inspection ({len(per_q)} questions)", expanded=False):
            st.caption(
                "Inspect retrieved context, generated answers, and per-question metric scores. "
                "Useful for demonstrating evaluation methodology."
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
                    "CtxRec":   _flt(r.get("context_recall")),
                    "LexF1":    f"{lf1.get('f1', 0):.2f}",
                    "Status":   "FAIL" if r.get("llm_error") else "OK",
                    "Question": r["question"][:55],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            for r in per_q:
                with st.expander(f"{r['id']} — {r['question'][:60]}", expanded=False):
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
                        st.text_area(f"Chunk {ci}", ctx[:400], height=80, disabled=True,
                                     key=f"ctx_{r['id']}_{ci}")

                    st.markdown("**Scores**")
                    lf1 = r.get("lexical_f1") or r.get("f1") or {}
                    sc  = st.columns(5)
                    sc[0].metric("Key Facts",    f"{r.get('key_fact_recall', 0):.2f}")
                    sc[1].metric("RHit",         "Y" if r.get("retrieval_recall_hit") else "N")
                    sc[2].metric("Faithfulness", _flt(r.get("faithfulness")))
                    sc[3].metric("Ans Rel",      _flt(r.get("answer_relevancy")))
                    sc[4].metric("Lex F1",       f"{lf1.get('f1', 0):.2f}")

    with st.expander("Metric methodology", expanded=False):
        methodology = report.get("metric_methodology", {})
        if methodology:
            for metric, desc in methodology.items():
                st.markdown(f"**`{metric}`** — {desc}")
        else:
            st.caption("No methodology documentation in this artifact.")


# Entry point

def main():
    _init()

    from config import Config
    if not Config.get_api_key() or Config.get_api_key() in (
        "hf_your_token_here", "sk-or-your-key-here", ""
    ):
        st.warning(
            "**No API key configured.** "
            "Add `HF_API_KEY` or `OPENROUTER_API_KEY` to your `.env` file to enable LLM and VLM."
        )

    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
