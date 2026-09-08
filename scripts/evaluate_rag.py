#!/usr/bin/env python3
# ruff: noqa: E402
"""
RAG Pipeline Evaluation Harness v2
====================================
Bounded, methodologically honest evaluation of the Multi-Model RAG pipeline.

Metric Tiers
------------
Tier 1 — Retrieval (deterministic, no LLM):
  - Retrieval Recall@K : fraction of questions where top-K chunks contain
                         the expected source section. Requires reference_section
                         labels. Honest retrieval quality signal.
  - Retrieval Token Overlap: token overlap between retrieved chunks and ground
                         truth answer. Diagnostic only — not "retrieval accuracy".

Tier 2 — Generation (LLM-as-Judge via ragas 0.2.x):
  - Faithfulness     : are claims in the answer supported by retrieved context?
                       (LLM decomposition + NLI check)
  - Answer Relevancy : does the answer address the question?
                       (embedding similarity of reverse-generated questions)
  - Context Recall   : does retrieved context contain ground-truth information?
                       (LLM-based classification per ground-truth sentence)

Tier 3 — Diagnostic Lexical (set-based, not ROUGE-1):
  - Key Fact Recall  : exact substring match for critical named entities/values
  - Lexical Precision/Recall/F1 : set-intersection token overlap
                         NOT equivalent to ROUGE-1 (which uses token counts).
                         Useful only to diagnose verbosity vs brevity.

API Cost Estimate (free tier)
------------------------------
  ragas uses ~2-4 LLM calls per question for Faithfulness + Context Recall,
  ~1 embedding call per question for Answer Relevancy.
  With 8 ragas questions × ~3 calls = ~24 LLM calls.
  Buffer: 5s sleep between questions, 2 retries max.

Usage
-----
    python scripts/evaluate_rag.py

Output
------
    artifacts/evaluation/rag_evaluation_results.json
"""

# ── Force UTF-8 output on Windows ─────────────────────────────────────────────
import sys as _sys
import io as _io
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
    _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── PyTorch / Windows DLL pre-init ────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _ST
    _ = _ST
except ImportError:
    pass

import sys
import os
import json
import time
import re
import tempfile
import statistics
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import DocumentPipeline  # noqa: E402
from config import Config  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation Document
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_DOCUMENT = (
    "CONSULTING AGREEMENT\n\n"
    "This Consulting Agreement (the 'Agreement') is entered into as of January 15, 2024, "
    "by and between Acme Corporation, a Delaware corporation with offices at 742 Evergreen "
    "Terrace, Springfield, IL 62704 ('Client'), and Jane Smith Consulting LLC, a California "
    "limited liability company ('Consultant').\n\n"
    "1. SCOPE OF SERVICES\n\n"
    "The Consultant agrees to provide strategic advisory services related to digital "
    "transformation, including but not limited to: (a) assessment of current technology "
    "infrastructure; (b) development of a three-year technology roadmap; (c) vendor "
    "evaluation and selection support; and (d) change management recommendations. The "
    "Consultant shall deliver a comprehensive written report within 90 days of the "
    "Effective Date.\n\n"
    "2. COMPENSATION\n\n"
    "Client shall pay Consultant a fixed fee of $84,200.00 for all services described "
    "herein. Payment shall be made in three installments: (i) $25,000 upon execution of "
    "this Agreement; (ii) $30,000 upon delivery of the interim report at Day 45; and "
    "(iii) $29,200 upon delivery of the final report. Late payments shall accrue interest "
    "at a rate of 1.5% per month.\n\n"
    "3. TERM AND TERMINATION\n\n"
    "This Agreement shall commence on the Effective Date and continue for a period of "
    "six (6) months unless earlier terminated. Either party may terminate this Agreement "
    "upon thirty (30) days' prior written notice to the other party. In the event of "
    "termination, the Consultant shall be compensated for all services performed up to "
    "the date of termination, calculated on a pro-rata basis.\n\n"
    "4. CONFIDENTIALITY\n\n"
    "Each party agrees to maintain the confidentiality of all proprietary information "
    "disclosed by the other party during the term of this Agreement. This obligation "
    "shall survive termination of this Agreement for a period of two (2) years. "
    "Confidential information includes, but is not limited to, trade secrets, customer "
    "lists, financial data, and business strategies.\n\n"
    "5. GOVERNING LAW\n\n"
    "This Agreement shall be governed by and construed in accordance with the laws of "
    "the State of Delaware, without regard to its conflict of laws provisions. Any "
    "disputes arising under this Agreement shall be resolved through binding arbitration "
    "in Wilmington, Delaware.\n\n"
    "IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first "
    "written above.\n\n"
    "Signed: John Doe, CEO, Acme Corporation\n"
    "Signed: Jane Smith, Managing Partner, Jane Smith Consulting LLC\n"
    "Contact: support@acme-corp.com | billing@janesmithconsulting.com\n"
    "Invoice Reference: INV-2024-0341\n"
)

# ─────────────────────────────────────────────────────────────────────────────
#  20-Question Benchmark
#
#  Extended from 15 → 20 to include harder, cross-section, and synthesis
#  questions that test genuine retrieval quality.
#
#  Fields:
#    id             : unique identifier
#    question       : natural language question
#    ground_truth   : correct, concise answer (traceable to document)
#    key_facts      : critical substrings that must appear in the answer
#    reference_section : source section keyword used for Retrieval Recall@K
#    ragas_eval     : whether to include in bounded ragas evaluation (cost control)
#    difficulty     : easy | medium | hard (honest self-assessment)
#    notes          : why this question tests a specific RAG failure mode
# ─────────────────────────────────────────────────────────────────────────────
EVALUATION_SET = [
    # ── Original 15 (kept unchanged, not tuned) ──────────────────────────────
    {
        "id": "Q01", "difficulty": "easy", "ragas_eval": True,
        "question": "What is the total fee amount for the consulting engagement?",
        "ground_truth": "The total fee is $84,200.00.",
        "key_facts": ["84,200", "$84,200"],
        "reference_section": "COMPENSATION",
        "notes": "Single numeric fact from COMPENSATION section",
    },
    {
        "id": "Q02", "difficulty": "medium", "ragas_eval": True,
        "question": "What are the three payment milestones and their amounts?",
        "ground_truth": "$25,000 upon execution, $30,000 at Day 45 interim report, $29,200 at final report.",
        "key_facts": ["25,000", "30,000", "29,200"],
        "reference_section": "COMPENSATION",
        "notes": "Multi-part numeric question — tests whether all three amounts are retrieved",
    },
    {
        "id": "Q03", "difficulty": "easy", "ragas_eval": False,
        "question": "Who are the two parties in this agreement?",
        "ground_truth": "Acme Corporation (Client) and Jane Smith Consulting LLC (Consultant).",
        "key_facts": ["Acme Corporation", "Jane Smith Consulting"],
        "reference_section": "CONSULTING AGREEMENT",
        "notes": "Entity recognition — appears in first paragraph",
    },
    {
        "id": "Q04", "difficulty": "easy", "ragas_eval": False,
        "question": "What is the late payment interest rate?",
        "ground_truth": "1.5% per month.",
        "key_facts": ["1.5%", "per month"],
        "reference_section": "COMPENSATION",
        "notes": "Single numeric fact",
    },
    {
        "id": "Q05", "difficulty": "medium", "ragas_eval": True,
        "question": "How can either party terminate this agreement early, and how much notice is required?",
        "ground_truth": "Either party may terminate upon thirty (30) days prior written notice to the other party.",
        "key_facts": ["thirty", "30", "written notice"],
        "reference_section": "TERM AND TERMINATION",
        "notes": "Termination procedure — tests TERM section retrieval",
    },
    {
        "id": "Q06", "difficulty": "medium", "ragas_eval": False,
        "question": "What happens to consultant fees if the agreement is terminated early?",
        "ground_truth": "The consultant is compensated for services performed up to termination on a pro-rata basis.",
        "key_facts": ["pro-rata", "termination"],
        "reference_section": "TERM AND TERMINATION",
        "notes": "Consequence question within TERM section",
    },
    {
        "id": "Q07", "difficulty": "easy", "ragas_eval": False,
        "question": "Which state's laws govern this agreement?",
        "ground_truth": "The State of Delaware.",
        "key_facts": ["Delaware"],
        "reference_section": "GOVERNING LAW",
        "notes": "Single entity from GOVERNING LAW section",
    },
    {
        "id": "Q08", "difficulty": "medium", "ragas_eval": True,
        "question": "For how long does the confidentiality obligation survive after the agreement ends?",
        "ground_truth": "Two years after termination of the agreement.",
        "key_facts": ["two", "2 years"],
        "reference_section": "CONFIDENTIALITY",
        "notes": "Post-termination duration — tests CONFIDENTIALITY section retrieval",
    },
    {
        "id": "Q09", "difficulty": "easy", "ragas_eval": False,
        "question": "What is the total duration of the consulting agreement?",
        "ground_truth": "Six months from the effective date.",
        "key_facts": ["six", "6 months"],
        "reference_section": "TERM AND TERMINATION",
        "notes": "Duration fact",
    },
    {
        "id": "Q10", "difficulty": "medium", "ragas_eval": True,
        "question": "What specific services is the consultant contracted to provide?",
        "ground_truth": (
            "Technology infrastructure assessment, three-year technology roadmap development, "
            "vendor evaluation and selection support, and change management recommendations."
        ),
        "key_facts": ["technology infrastructure", "roadmap", "vendor evaluation", "change management"],
        "reference_section": "SCOPE OF SERVICES",
        "notes": "Multi-item scope question — tests whether all four services are retrieved",
    },
    {
        "id": "Q11", "difficulty": "easy", "ragas_eval": False,
        "question": "What is the invoice reference number in this agreement?",
        "ground_truth": "INV-2024-0341.",
        "key_facts": ["INV-2024-0341"],
        "reference_section": "Invoice Reference",
        "notes": "Exact ID lookup from signature block",
    },
    {
        "id": "Q12", "difficulty": "medium", "ragas_eval": False,
        "question": "Where and how will any disputes under this agreement be resolved?",
        "ground_truth": "Through binding arbitration in Wilmington, Delaware.",
        "key_facts": ["arbitration", "Wilmington", "Delaware"],
        "reference_section": "GOVERNING LAW",
        "notes": "Dispute resolution — two facts from GOVERNING LAW section",
    },
    {
        "id": "Q13", "difficulty": "easy", "ragas_eval": False,
        "question": "What is the effective date of this consulting agreement?",
        "ground_truth": "January 15, 2024.",
        "key_facts": ["January 15, 2024"],
        "reference_section": "CONSULTING AGREEMENT",
        "notes": "Date extraction from opening paragraph",
    },
    {
        "id": "Q14", "difficulty": "easy", "ragas_eval": False,
        "question": "What is Acme Corporation's registered office address?",
        "ground_truth": "742 Evergreen Terrace, Springfield, IL 62704.",
        "key_facts": ["742 Evergreen", "Springfield", "62704"],
        "reference_section": "CONSULTING AGREEMENT",
        "notes": "Full address — multiple address tokens in one chunk",
    },
    {
        "id": "Q15", "difficulty": "easy", "ragas_eval": False,
        "question": "Within how many days of the effective date must the consultant deliver the final report?",
        "ground_truth": "Within 90 days of the effective date.",
        "key_facts": ["90 days"],
        "reference_section": "SCOPE OF SERVICES",
        "notes": "Deadline from SCOPE section",
    },
    # ── 5 Harder Questions (cross-section, synthesis, careful reading) ────────
    {
        "id": "Q16", "difficulty": "hard", "ragas_eval": True,
        "question": (
            "If the client terminates the agreement at Day 30, "
            "how much is the consultant owed based on the payment schedule?"
        ),
        "ground_truth": (
            "The consultant is owed a pro-rata amount for 30 out of 180 days. "
            "At the $84,200 total fee, that is approximately $14,033 (30/180 × $84,200). "
            "Only the $25,000 execution payment would be due at Day 30; "
            "the Day 45 and final milestones would not yet be triggered."
        ),
        "key_facts": ["pro-rata", "25,000", "84,200"],
        "reference_section": ["TERM AND TERMINATION", "COMPENSATION"],
        "notes": "Cross-section synthesis: COMPENSATION + TERM. Tests whether LLM can combine payment schedule with termination clause. Requires reasoning, not just extraction.",
    },
    {
        "id": "Q17", "difficulty": "hard", "ragas_eval": True,
        "question": (
            "What categories of information are explicitly listed as confidential "
            "under this agreement?"
        ),
        "ground_truth": (
            "Trade secrets, customer lists, financial data, and business strategies."
        ),
        "key_facts": ["trade secrets", "customer lists", "financial data", "business strategies"],
        "reference_section": "CONFIDENTIALITY",
        "notes": "Multi-item extraction from CONFIDENTIALITY section. Tests whether all four are retrieved and included.",
    },
    {
        "id": "Q18", "difficulty": "hard", "ragas_eval": True,
        "question": (
            "What is the total amount of the second and third payment milestones combined?"
        ),
        "ground_truth": (
            "The second milestone is $30,000 (Day 45 interim report) and the third is $29,200 "
            "(final report), totalling $59,200."
        ),
        "key_facts": ["30,000", "29,200", "59,200"],
        "reference_section": "COMPENSATION",
        "notes": "Numeric calculation question — requires the LLM to add two values from context, not just extract them.",
    },
    {
        "id": "Q19", "difficulty": "hard", "ragas_eval": False,
        "question": (
            "Under what circumstances does the confidentiality obligation continue "
            "after the agreement period ends, and for how long?"
        ),
        "ground_truth": (
            "The confidentiality obligation survives termination of the agreement "
            "for a period of two years, regardless of how the agreement ends."
        ),
        "key_facts": ["two years", "survive", "termination"],
        "reference_section": "CONFIDENTIALITY",
        "notes": "Conditional/nuanced question. Tests whether the LLM correctly identifies 'survive termination' language.",
    },
    {
        "id": "Q20", "difficulty": "hard", "ragas_eval": False,
        "question": (
            "Is the consultant an employee of Acme Corporation under this agreement?"
        ),
        "ground_truth": (
            "The agreement does not explicitly classify the consultant as an employee or independent contractor. "
            "However, Jane Smith Consulting LLC is described as the 'Consultant', not an employee, "
            "and the structure of payment milestones and deliverables suggests an independent contractor relationship."
        ),
        "key_facts": ["Consultant", "Jane Smith Consulting"],
        "reference_section": "CONSULTING AGREEMENT",
        "notes": "Inference/negative question. The document never says 'independent contractor' explicitly — tests whether the LLM correctly avoids hallucinating employment status.",
    },
]

RAGAS_SUBSET = [item for item in EVALUATION_SET if item.get("ragas_eval", False)]


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 1: Deterministic Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set:
    """Lowercase, strip punctuation, return unique token set."""
    return set(re.sub(r"[^\w\s]", "", text.lower()).split())


def compute_lexical_f1(prediction: str, ground_truth: str) -> dict:
    """
    Set-based token-level Precision, Recall, F1.

    IMPORTANT: This is NOT ROUGE-1. ROUGE-1 uses token COUNTS (allowing
    duplicates). This uses set intersection, so duplicate tokens are collapsed.
    Use for diagnostic purposes only — high recall is expected when the LLM
    includes all ground-truth terms plus extra words (verbosity effect).
    """
    pred_tokens  = _tokenize(prediction)
    truth_tokens = _tokenize(ground_truth)
    if not pred_tokens or not truth_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    common    = pred_tokens & truth_tokens
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(truth_tokens)
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


def compute_key_fact_recall(answer: str, key_facts: list) -> float:
    """
    Exact substring match for critical facts.
    Valid metric: if a required entity/number is absent, the answer is wrong.
    This is resistant to the verbosity problem that inflates lexical recall.
    """
    if not key_facts:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for fact in key_facts if fact.lower() in answer_lower)
    return round(found / len(key_facts), 4)


def compute_retrieval_token_overlap(retrieved_texts: list, ground_truth: str) -> float:
    """
    Token overlap between retrieved chunks and ground-truth answer.
    Diagnostic signal: if overlap is high, retriever found the right passage.
    NOT the same as Retrieval Recall — does not verify source labels.
    """
    if not retrieved_texts:
        return 0.0
    combined     = " ".join(retrieved_texts)
    truth_tokens = _tokenize(ground_truth)
    ctx_tokens   = _tokenize(combined)
    if not truth_tokens:
        return 0.0
    return round(len(truth_tokens & ctx_tokens) / len(truth_tokens), 4)


def compute_retrieval_recall_at_k(retrieved_texts: list, reference_section: str | list) -> bool:
    """
    Retrieval Recall@K — deterministic, no LLM.

    Returns True if the top-K retrieved chunks contain ALL the required
    reference_section keywords. For synthesis questions, this strictly verifies
    that all pieces of necessary context were retrieved.
    """
    if not retrieved_texts or not reference_section:
        return False

    sections = [reference_section] if isinstance(reference_section, str) else reference_section

    for sec in sections:
        sec_lower = sec.lower()
        # If any required section is completely missing from all retrieved chunks, we fail
        if not any(sec_lower in t.lower() for t in retrieved_texts):
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Tier 2: ragas (LLM-as-Judge) — bounded evaluation
# ─────────────────────────────────────────────────────────────────────────────

def build_ragas_evaluator():
    """
    Build a ragas evaluator using the project's existing HF/OpenRouter endpoint.
    Returns (evaluate_fn, ragas_llm, ragas_embeddings) or None if unavailable.
    """
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os

        langchain_llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            temperature=0,
        )
        ragas_llm = LangchainLLMWrapper(langchain_llm)

        # Use local sentence-transformers for answer relevancy embeddings
        # (no API calls needed for embeddings)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        local_emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        ragas_emb = LangchainEmbeddingsWrapper(local_emb)

        print("  ragas LLM wrapper:        OK (Gemini endpoint)")
        print("  ragas embeddings wrapper: OK (local all-MiniLM-L6-v2)")
        return ragas_llm, ragas_emb

    except Exception as exc:
        print(f"  ragas setup failed: {exc}")
        return None, None


def run_ragas_evaluation(ragas_subset, ragas_llm, ragas_emb) -> dict:
    """
    Run ragas Faithfulness, Answer Relevancy, and Context Recall
    on a bounded subset of questions.

    Faithfulness   : LLM decomposes answer into statements, then checks
                     each against retrieved context. Detects hallucination.
    Answer Relevancy: Embedding similarity between question and LLM-generated
                     reverse questions from the answer. Detects irrelevant answers.
    Context Recall : LLM checks whether each ground-truth sentence is attributable
                     to the retrieved context. Measures retrieval completeness.

    API cost: ~3 LLM calls per question + 1 embedding call per question.
    With 8 questions: ~24 LLM calls, ~8 embedding calls.
    """
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall

    # Configure metrics with our LLM and embeddings
    faithfulness.llm    = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_emb
    context_recall.llm  = ragas_llm

    # Build dataset
    questions      = [item["question"]     for item in ragas_subset]
    ground_truths  = [item["ground_truth"] for item in ragas_subset]
    answers        = [item["_answer"]      for item in ragas_subset]
    contexts_list  = [item["_contexts"]    for item in ragas_subset]

    eval_data = {
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(eval_data)

    result = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        raise_exceptions=False,
    )

    df = result.to_pandas()

    per_q = []
    for i, item in enumerate(ragas_subset):
        row = df.iloc[i] if i < len(df) else {}
        per_q.append({
            "id":               item["id"],
            "faithfulness":     _safe_float(row.get("faithfulness")),
            "answer_relevancy": _safe_float(row.get("answer_relevancy")),
            "context_recall":   _safe_float(row.get("context_recall")),
        })

    return {
        "faithfulness":     _safe_float(df["faithfulness"].mean()     if "faithfulness"     in df else None),
        "answer_relevancy": _safe_float(df["answer_relevancy"].mean() if "answer_relevancy" in df else None),
        "context_recall":   _safe_float(df["context_recall"].mean()   if "context_recall"   in df else None),
        "num_evaluated":    len(ragas_subset),
        "per_question":     per_q,
    }


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return round(f, 4) if not (f != f) else None  # NaN check
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Main Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("\n" + "=" * 66)
    print("  RAG Pipeline Evaluation Harness v2")
    print("=" * 66)
    print(
        "\n  Benchmark: 20 questions (15 original + 5 harder synthesis/inference)\n"
        "  ragas evaluation: 8 bounded questions (free-tier API cost control)\n"
        "  All scores computed from live pipeline — none are hardcoded.\n"
    )

    # ── 1. Write document ─────────────────────────────────────────────────
    tmp_dir  = tempfile.mkdtemp(prefix="rag_eval_v2_")
    tmp_file = os.path.join(tmp_dir, "consulting_agreement.txt")
    with open(tmp_file, "w", encoding="utf-8") as fh:
        fh.write(SAMPLE_DOCUMENT)
    print(f"  Document written: {len(SAMPLE_DOCUMENT):,} chars\n")

    # ── 2. Init pipeline ──────────────────────────────────────────────────
    print("  Initialising pipeline...")
    pipeline = DocumentPipeline()
    print("  Pipeline ready\n")

    # ── 3. Index ──────────────────────────────────────────────────────────
    collection_name = "consulting_agreement.txt"
    print("  Indexing document...")
    idx = pipeline.index(tmp_file, original_filename=collection_name)
    print(f"  Indexed {idx['total_chunks']} chunks in {idx['processing_time_sec']}s\n")

    # ── 4. ragas setup ────────────────────────────────────────────────────
    print("  Setting up ragas evaluator...")
    ragas_llm, ragas_emb = build_ragas_evaluator()
    ragas_available = ragas_llm is not None
    print()

    # ── 5. Main evaluation loop ───────────────────────────────────────────
    results       = []
    total         = len(EVALUATION_SET)
    succeeded_llm = 0
    failed_llm    = 0

    from src.retrieval.hybrid_retriever import HybridRetriever

    for i, item in enumerate(EVALUATION_SET, 1):
        qid = item["id"]
        q   = item["question"]
        gt  = item["ground_truth"]
        kf  = item["key_facts"]
        ref = item["reference_section"]

        q_short = q[:52] + "…" if len(q) > 52 else q
        diff    = item.get("difficulty", "?")
        print(f"  [{i:>2}/{total}] {qid} ({diff}): {q_short}")

        # ── LLM answer ───────────────────────────────────────────────
        answer    = ""
        llm_error = None

        for attempt in range(1, 3):  # max 2 attempts
            try:
                response = pipeline.query(q, collection_name=collection_name)
                answer   = response.get("answer", "")
                error_phrases = [
                    "api authentication failed", "rate limit",
                    "no documents have been indexed", "llm error",
                ]
                if any(p in answer.lower() for p in error_phrases):
                    raise RuntimeError(f"LLM error response: {answer[:80]}")
                succeeded_llm += 1
                break
            except Exception as exc:
                llm_error = str(exc)
                if attempt < 2:
                    print(f"         Attempt {attempt} failed. Retry in 6s...")
                    time.sleep(6)
                else:
                    print(f"         All attempts failed — skipping LLM for {qid}")
                    failed_llm += 1

        # ── Retrieve chunks for metrics ───────────────────────────────
        retrieved_texts    = []
        try:
            chunks = pipeline.all_chunks.get(collection_name, [])
            retriever = HybridRetriever(embedder=pipeline.embedder, chunks=chunks)
            retrieved_chunks = retriever.retrieve(query=q, collection_name=collection_name)
            retrieved_texts    = [r["text"] for r in retrieved_chunks]
        except Exception as exc:
            print(f"         Retrieval error: {exc}")

        # ── Tier 1 metrics ────────────────────────────────────────────
        lex_f1  = compute_lexical_f1(answer, gt) if answer else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        kfr     = compute_key_fact_recall(answer, kf) if answer else 0.0
        rto     = compute_retrieval_token_overlap(retrieved_texts, gt)
        ret_hit = compute_retrieval_recall_at_k(retrieved_texts, ref)

        result_entry = {
            "id":                    qid,
            "difficulty":            diff,
            "question":              q,
            "ground_truth":          gt,
            "reference_section":     ref,
            "answer":                answer,
            "llm_error":             llm_error,
            "retrieved_contexts":    retrieved_texts,
            "retrieved_count":       len(retrieved_texts),
            "retrieval_recall_hit":  ret_hit,
            "retrieval_token_overlap": rto,
            "key_fact_recall":       kfr,
            "lexical_f1":            lex_f1,
            # ragas filled in later
            "faithfulness":          None,
            "answer_relevancy":      None,
            "context_recall":        None,
            # for ragas batch processing
            "_answer":   answer,
            "_contexts": retrieved_texts,
        }
        results.append(result_entry)

        status = "OK" if not llm_error else "FAIL"
        print(
            f"         [{status}] KFR={kfr:.2f}  RHit={'Y' if ret_hit else 'N'}"
            f"  RTO={rto:.2f}  F1={lex_f1['f1']:.2f}"
        )
        time.sleep(4)  # rate-limit buffer

    # ── 6. ragas evaluation (bounded subset) ──────────────────────────────
    ragas_aggregate = {
        "faithfulness":     None,
        "answer_relevancy": None,
        "context_recall":   None,
        "num_evaluated":    0,
        "status":           "not_attempted",
    }

    if ragas_available:
        # Only evaluate questions where LLM succeeded AND ragas_eval=True
        ragas_items = [
            r for r in results
            if not r["llm_error"] and any(
                e["id"] == r["id"] and e.get("ragas_eval", False)
                for e in EVALUATION_SET
            )
        ]
        if ragas_items:
            print(f"\n  Running ragas on {len(ragas_items)} questions...")
            print("  (Faithfulness + Answer Relevancy + Context Recall)")
            try:
                ragas_result = run_ragas_evaluation(ragas_items, ragas_llm, ragas_emb)
                ragas_aggregate = {**ragas_result, "status": "completed"}

                # Backfill per-question ragas scores
                ragas_by_id = {r["id"]: r for r in ragas_result.get("per_question", [])}
                for entry in results:
                    if entry["id"] in ragas_by_id:
                        rq = ragas_by_id[entry["id"]]
                        entry["faithfulness"]      = rq.get("faithfulness")
                        entry["answer_relevancy"]  = rq.get("answer_relevancy")
                        entry["context_recall"]    = rq.get("context_recall")
                print(f"  ragas complete: Faithfulness={ragas_aggregate['faithfulness']}")
            except Exception as exc:
                ragas_aggregate["status"] = f"failed: {exc}"
                print(f"  ragas evaluation failed: {exc}")
        else:
            ragas_aggregate["status"] = "skipped — no eligible questions"
    else:
        ragas_aggregate["status"] = "unavailable — ragas LLM setup failed"

    # ── 7. Aggregate Tier 1 ───────────────────────────────────────────────
    answered = [r for r in results if not r["llm_error"]]

    def _mean(vals):
        clean = [v for v in vals if v is not None]
        return round(statistics.mean(clean), 4) if clean else None

    def _std(vals):
        clean = [v for v in vals if v is not None]
        return round(statistics.stdev(clean), 4) if len(clean) > 1 else 0.0

    kfr_vals = [r["key_fact_recall"]          for r in answered]
    rto_vals = [r["retrieval_token_overlap"]   for r in results]
    rr_vals  = [1.0 if r["retrieval_recall_hit"] else 0.0 for r in results]
    f1_vals  = [r["lexical_f1"]["f1"]          for r in answered]
    p_vals   = [r["lexical_f1"]["precision"]   for r in answered]
    rec_vals = [r["lexical_f1"]["recall"]      for r in answered]

    tier1 = {
        "retrieval_recall_at_k":         _mean(rr_vals),
        "retrieval_recall_at_k_std":     _std(rr_vals),
        "retrieval_token_overlap":       _mean(rto_vals),
        "retrieval_token_overlap_std":   _std(rto_vals),
        "key_fact_recall":               _mean(kfr_vals),
        "key_fact_recall_std":           _std(kfr_vals),
        "lexical_f1":                    _mean(f1_vals),
        "lexical_f1_std":                _std(f1_vals),
        "lexical_precision":             _mean(p_vals),
        "lexical_recall":                _mean(rec_vals),
    }

    # ── 8. Console report ─────────────────────────────────────────────────
    n = len(results)
    print("\n" + "=" * 66)
    print("  EVALUATION RESULTS")
    print("=" * 66)
    print(f"  Total questions   : {n}")
    print(f"  LLM answers OK    : {succeeded_llm}")
    print(f"  LLM answers FAILED: {failed_llm}")
    print(f"  ragas status      : {ragas_aggregate['status']}")
    print()
    print("  TIER 1 — Retrieval (deterministic)")
    print(f"    Retrieval Recall@K  : {tier1['retrieval_recall_at_k']:.4f} (section keyword in top-5 chunks)")
    print(f"    Retrieval Tok Overlap: {tier1['retrieval_token_overlap']:.4f} (diagnostic, not 'accuracy')")
    print()
    print("  TIER 1 — Generation (deterministic)")
    print(f"    Key Fact Recall     : {tier1['key_fact_recall']:.4f} (exact match for critical entities)")
    print(f"    Lexical F1 (not ROUGE-1): {tier1['lexical_f1']:.4f}")
    print()
    print("  TIER 2 — LLM-as-Judge (ragas 0.2.x)")
    for metric in ("faithfulness", "answer_relevancy", "context_recall"):
        val = ragas_aggregate.get(metric)
        tag = f"{val:.4f} (n={ragas_aggregate['num_evaluated']})" if val is not None else "N/A — not evaluated"
        print(f"    {metric:<20}: {tag}")
    print()
    print("  Per-Question (Retrieval Recall@K | Key Facts | ragas Faithfulness):")
    for r in results:
        rh = "Y" if r["retrieval_recall_hit"] else "N"
        kf = f"{r['key_fact_recall']:.2f}"
        fa = f"{r['faithfulness']:.2f}" if r["faithfulness"] is not None else " N/A"
        er = " FAIL" if r["llm_error"] else "   OK"
        print(f"    {r['id']} ({r['difficulty']:>6}) [{er}]  RHit={rh}  KFR={kf}  Faith={fa}  | {r['question'][:40]}")
    print("=" * 66)

    # ── 9. Save JSON artifact ─────────────────────────────────────────────
    project_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    artifact_dir  = os.path.join(project_root, "artifacts", "evaluation")
    os.makedirs(artifact_dir, exist_ok=True)
    artifact_path = os.path.join(artifact_dir, "rag_evaluation_results.json")

    # Remove internal-only fields before saving
    clean_results = []
    for r in results:
        clean = {k: v for k, v in r.items() if not k.startswith("_")}
        clean_results.append(clean)

    report = {
        "schema_version":        "2.0",
        "evaluation_timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "document":              collection_name,
        "model_backend":         Config.get_backend_name(),
        "llm_model":             Config.get_llm_model(),
        "num_questions":         n,
        "llm_answers_succeeded": succeeded_llm,
        "llm_answers_failed":    failed_llm,
        "metric_methodology": {
            "retrieval_recall_at_k":    (
                "Retrieval Recall@K (deterministic): 1 if ANY of the top-K retrieved chunks "
                "contains the reference_section keyword, else 0. Keyword heuristic — not semantic."
            ),
            "retrieval_token_overlap":  (
                "Diagnostic only: token-set overlap between retrieved chunks and ground-truth answer. "
                "High overlap = retriever found the right passage. NOT 'retrieval accuracy'."
            ),
            "key_fact_recall":          (
                "Exact substring match for critical named entities, numbers, and dates. "
                "Fraction of key_facts found in the answer. Zero tolerance for missing facts."
            ),
            "lexical_f1":               (
                "Set-based token overlap F1. NOT ROUGE-1 (which uses token counts allowing duplicates). "
                "Inflated by LLM verbosity. Use Key Fact Recall as primary quality signal instead."
            ),
            "faithfulness":             (
                "ragas 0.2.x Faithfulness (LLM-as-Judge): decomposes answer into statements, "
                "verifies each against retrieved context using NLI. Measures hallucination resistance."
            ),
            "answer_relevancy":         (
                "ragas 0.2.x Answer Relevancy: generates reverse questions from answer, "
                "measures embedding similarity to original question. Detects irrelevant answers."
            ),
            "context_recall":           (
                "ragas 0.2.x Context Recall: classifies each ground-truth sentence as attributable "
                "or not attributable to retrieved context. Measures retrieval completeness."
            ),
        },
        "tier1_aggregate":       tier1,
        "tier2_ragas":           ragas_aggregate,
        "per_question_results":  clean_results,
    }

    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"\n  Artifact saved: {artifact_path}")

    # Cleanup
    try:
        os.unlink(tmp_file)
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print(f"  Done: {n} questions, {succeeded_llm} answered, {failed_llm} failed.\n")
    return report


if __name__ == "__main__":
    run_evaluation()
