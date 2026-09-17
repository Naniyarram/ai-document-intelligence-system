
# src/generation/llm_handler.py

# STAGE 6 — LLM Generation

import re
from typing import List
from loguru import logger
from config import Config


SYSTEM_PROMPT = """You are an intelligent document analysis assistant.

Your job is to answer questions based STRICTLY on the document context provided.

Rules:
1. ONLY use information from the provided context. Do not use external knowledge or hallucinate facts.
2. If the answer is not in the context, say clearly: "The document does not provide enough information to answer this."
3. For tables, numbers, and dates, reproduce exact values — do not estimate or infer.
4. Keep answers concise and directly address the user's specific question. Do not summarize the entire document unless asked.
5. Do not infer, embellish, or draw conclusions that are not explicitly supported by the text.
6. For follow-up questions, use the conversation history to understand references like "it" or "that".
7. Do not append trailing 'Source:' lines or citation tags in your text response (citations are managed separately).

Format:
- Adapt your structure to the question: use a short paragraph for explanations, bullet points for lists/summaries, and structured text for data extraction.
- Avoid conversational filler (e.g., "Based on the provided context..."). Start your answer directly.
"""


class LLMHandler:
    """
    Handles all LLM calls for document Q&A.

    Maintains conversation history so users can ask follow-up questions.
    Automatically uses HuggingFace or OpenRouter based on the API key type.

    Usage:
        llm = LLMHandler()
        answer = llm.answer("What are the payment terms?", retrieved_chunks)
        follow_up = llm.answer("And the penalties?", new_chunks)
    """

    def __init__(self):
        from openai import OpenAI
        import httpx

        api_key  = Config.get_api_key()
        base_url = Config.get_base_url()
        backend  = Config.get_backend_name()

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(
                trust_env=False,
                timeout=httpx.Timeout(60.0, connect=20.0),
            ),
        )
        self.model  = Config.get_llm_model()

        # Conversation history — enables follow-up questions
        self.conversation_history: List[dict] = []

        logger.info(f"LLM Handler ready (backend={backend}, model={self.model})")

    def answer(
        self,
        query: str,
        retrieved_chunks: List[dict],
        mode: str = "qa",
    ) -> dict:
        """
        Generate an answer using the retrieved context chunks.

        Args:
            query:            The user's question
            retrieved_chunks: Top-k chunks from hybrid retrieval
            mode:             "qa" | "extract" | "summarize" | "anomaly"

        Returns:
            dict: {answer, sources, context_used, model}
        """
        if not retrieved_chunks:
            return {
                "answer": "I couldn't find relevant information in the document.",
                "sources": [],
                "context_used": 0,
            }

        # Format chunks as a readable context block
        context = self._format_context(retrieved_chunks)

        # Build the user message (includes context + question)
        user_message = self._build_user_message(query, context, mode)

        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})

        # Full messages list: system prompt + all conversation turns
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation_history

        try:
            answer_text = self._generate_with_fallback(messages)

            # Add assistant reply to history (enables follow-up questions)
            self.conversation_history.append({"role": "assistant", "content": answer_text})

            # Keep history to last 10 exchanges (20 messages)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            return {
                "answer":       answer_text,
                "sources":      self._extract_sources(retrieved_chunks),
                "context_used": len(retrieved_chunks),
                "model":        self.model,
                "quality":      self._estimate_answer_quality(answer_text, retrieved_chunks),
            }

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            fallback_answer = (
                self._build_grounded_fallback_answer(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    mode=mode,
                )
                if Config.ENABLE_LOCAL_LLM_FALLBACK
                else ""
            )
            if fallback_answer and not self._is_invalid_llm_response(fallback_answer):
                return {
                    "answer": fallback_answer,
                    "sources": self._extract_sources(retrieved_chunks),
                    "context_used": len(retrieved_chunks),
                    "model": "local-fallback",
                    "quality": self._estimate_answer_quality(
                        fallback_answer,
                        retrieved_chunks,
                        used_fallback=True,
                    ),
                    "is_error": False,
                }

            # Give a helpful error message based on error type
            err = str(e).lower()
            if "429" in err or "rate" in err:
                msg = (
                    "The LLM provider is currently rate-limited (429 Rate Limit) or unavailable. "
                    "Retrieval succeeded, but generation could not complete. Please wait a moment and try again."
                )
            elif "401" in err or "unauthorized" in err or "authentication" in err:
                msg = (
                    "API authentication failed. Please check your token in .env:\n"
                    "- HuggingFace token: set HF_API_KEY=hf_your_token\n"
                    "- Make sure the token has 'Make calls to Inference Providers' permission\n"
                    f"- Current backend: {Config.get_backend_name()}"
                )
            elif "404" in err or "not found" in err or "model" in err:
                msg = (
                    f"Model '{self.model}' not available.\n"
                    f"Change LLM_MODEL in your .env file or leave it blank for auto-selection.\n"
                    f"Recommended: LLM_MODEL={Config.get_llm_model()}"
                )
            elif "connection" in err or "timeout" in err or "network" in err:
                msg = (
                    "The LLM provider could not be reached.\n"
                    "Please check your internet connection and confirm the API provider is accessible.\n"
                    f"Current backend: {Config.get_backend_name()}"
                )
            else:
                msg = f"LLM generation unavailable: {str(e)}"

            return {
                "answer": msg,
                "sources": self._extract_sources(retrieved_chunks),
                "context_used": len(retrieved_chunks),
                "is_error": True,
                "model": self.model,
            }

    def _build_grounded_fallback_answer(
        self,
        query: str,
        retrieved_chunks: List[dict],
        mode: str,
    ) -> str:
        """
        Build a concise, synthesized answer from retrieved chunks when the remote
        LLM is unavailable. Keeps the response grounded in the document.
        """
        if not retrieved_chunks:
            return ""

        if self._is_document_overview_query(query, mode):
            overview = self._build_document_overview(retrieved_chunks)
            if overview:
                return overview

        query_terms = {
            token for token in re.findall(r"\b\w+\b", query.lower())
            if len(token) > 2
        }

        ranked_sentences = []
        for chunk in retrieved_chunks[:4]:
            text = chunk.get("text", "")
            sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
            for sentence in sentences:
                cleaned = sentence.strip()
                cleaned = re.sub(r"^\[?\d+[\.\s]+[A-Z\s]+\]?", "", cleaned).strip()
                if not cleaned or len(cleaned) < 20:
                    continue
                sentence_terms = set(re.findall(r"\b\w+\b", cleaned.lower()))
                overlap = len(query_terms & sentence_terms)
                ranked_sentences.append((overlap, cleaned))

        ranked_sentences.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        selected = []
        seen = set()
        for _, sentence in ranked_sentences:
            key = sentence.lower()
            if key not in seen:
                seen.add(key)
                selected.append(sentence)
            if len(selected) >= 4:
                break

        if not selected:
            return ""

        lines = []
        for sentence in selected:
            lines.append(f"• {sentence}")

        return "\n".join(lines)

    def _is_document_overview_query(self, query: str, mode: str) -> bool:
        """Detect broad 'what is this document' style questions."""
        if mode == "summarize":
            return True

        query_lower = query.lower()
        overview_phrases = [
            "explain the content",
            "explain content",
            "what is this document",
            "what does this document contain",
            "summarize this document",
            "summary of this document",
            "describe the document",
            "overview of the document",
        ]
        return any(phrase in query_lower for phrase in overview_phrases)

    def _build_document_overview(self, retrieved_chunks: List[dict]) -> str:
        """Create a clean, synthesized document summary from retrieved chunks."""
        combined = "\n".join(chunk.get("text", "") for chunk in retrieved_chunks)
        if not combined.strip():
            return ""

        first_meta = retrieved_chunks[0].get("metadata", {})
        doc_name = first_meta.get("source_file", "Document")

        raw_lines = [line.strip() for line in combined.splitlines() if line.strip()]
        bullets = []
        seen = set()

        for line in raw_lines:
            cleaned = re.sub(r"^\[?\d+[\.\s]+[A-Z\s]+\]?", "", line).strip()
            if len(cleaned) < 18 or cleaned.isupper():
                continue
            if len(cleaned) > 220:
                cleaned = cleaned[:220].rsplit(" ", 1)[0] + "..."

            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                bullets.append(f"• {cleaned}")
            if len(bullets) >= 5:
                break

        if not bullets:
            return f"Summary of {doc_name} based on retrieved context."

        return f"Summary of {doc_name}:\n\n" + "\n".join(bullets)

    def _estimate_answer_quality(
        self,
        answer_text: str,
        retrieved_chunks: List[dict],
        used_fallback: bool = False,
    ) -> dict:
        """Estimate simple confidence and faithfulness scores from local context."""
        if not answer_text or not retrieved_chunks:
            return {"confidence": 0.0, "faithfulness": 0.0}

        context = " ".join(chunk.get("text", "") for chunk in retrieved_chunks).lower()
        context_terms = set(re.findall(r"\b\w+\b", context))
        answer_terms = [
            token for token in re.findall(r"\b\w+\b", answer_text.lower())
            if len(token) > 2 and token not in {"source", "document", "page"}
        ]

        if answer_terms:
            supported_terms = sum(1 for token in answer_terms if token in context_terms)
            lexical_support = supported_terms / len(answer_terms)
        else:
            lexical_support = 0.5

        retrieval_scores = [float(chunk.get("score", 0.0)) for chunk in retrieved_chunks]
        avg_retrieval = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0

        confidence = min(0.99, max(0.05, (0.55 * lexical_support) + (0.45 * avg_retrieval)))
        faithfulness = min(0.99, max(0.05, lexical_support))

        if used_fallback:
            confidence = max(0.05, confidence - 0.08)

        return {
            "confidence": round(confidence, 2),
            "faithfulness": round(faithfulness, 2),
        }

    def _candidate_models(self) -> List[str]:
        """Return the configured model plus backend-specific fallbacks."""
        ordered = [self.model] + Config.get_llm_fallback_models()
        unique = []
        for model_name in ordered:
            if model_name and model_name not in unique:
                unique.append(model_name)
        return unique

    def _generate_with_fallback(self, messages: List[dict]) -> str:
        """
        Try the configured LLM first, then retry with safe fallbacks if the
        provider reports that a model is unavailable.
        """
        last_error = None
        for model_name in self._candidate_models():
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.1,
                )
                if model_name != self.model:
                    logger.warning(
                        f"Configured LLM '{self.model}' unavailable; switched to '{model_name}'"
                    )
                    self.model = model_name
                if not getattr(response, "choices", None):
                    raise ValueError(f"Provider returned no choices for model {model_name}")

                content = response.choices[0].message.content
                if content is None:
                    raise ValueError(f"Provider returned empty content for model {model_name}")

                return content.strip()
            except Exception as exc:
                last_error = exc
                if not self._is_model_unavailable_error(exc):
                    raise
                logger.warning(f"LLM model '{model_name}' unavailable: {exc}")

        raise last_error

    @staticmethod
    def _is_model_unavailable_error(error: Exception) -> bool:
        """Identify provider errors that mean the model ID is unsupported or rate limited."""
        message = str(error).lower()
        return any(
            clue in message
            for clue in ("404", "429", "model", "not found", "does not exist", "unavailable", "rate", "upstream")
        )

    def reset_conversation(self):
        """Clear conversation history (call when switching documents)."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def _format_context(self, chunks: List[dict]) -> str:
        """Format retrieved chunks into a labelled context block."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            meta    = chunk.get("metadata", {})
            source  = meta.get("source_file", "Unknown")
            page    = meta.get("page_number", "?")
            section = meta.get("section_title", "")

            label = f"[Source {i}: {source}, Page {page}"
            if section:
                label += f", Section: {section}"
            label += "]"

            parts.append(f"{label}\n{chunk['text']}")

        return "\n\n---\n\n".join(parts)

    def _build_user_message(self, query: str, context: str, mode: str) -> str:
        """Build the prompt with context and task-specific instruction."""
        instructions = {
            "qa":       "Answer the question based only on the context above.",
            "extract":  "Extract the requested information from the context above. Format structured data as a clear list or table.",
            "summarize":"Provide a concise summary of the context above. Focus on key points.",
            "anomaly":  "Analyze the data for anomalies, outliers, duplicates, or unusual patterns. Be specific.",
        }
        instruction = instructions.get(mode, instructions["qa"])

        if self._is_document_overview_query(query, mode):
            instruction = (
                "Provide a clean document overview using only the context above. "
                "State the document type first, then summarize the most important "
                "identifiers, parties, dates, totals, items, and notable notes in "
                "5 to 7 short bullet points. Avoid irrelevant boilerplate."
            )

        return f"""Context from the document:

{context}

---

{instruction}

Question: {query}"""

    def _extract_sources(self, chunks: List[dict]) -> List[dict]:
        """Extract unique source references from retrieved chunks."""
        seen    = set()
        sources = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            key  = f"{meta.get('source_file', '')}_{meta.get('page_number', '')}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file":         meta.get("source_file", "Unknown"),
                    "page":         meta.get("page_number", "?"),
                    "section":      meta.get("section_title", ""),
                    "content_type": meta.get("content_type", "text"),
                })
        return sources
