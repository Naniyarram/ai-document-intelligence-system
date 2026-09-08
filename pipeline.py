# pipeline.py — Main Pipeline Orchestrator
#
# Connects all 6 stages:
#   1. Ingestion   -> document_loader.py
#   2. VLM         -> vlm_handler.py
#   3. Chunking    -> semantic_chunker.py  (LangChain)
#   4. Embedding   -> embedder.py          (ChromaDB)
#   5. Retrieval   -> hybrid_retriever.py  (BM25 + Dense + Reranker)
#   6. Generation  -> llm_handler.py       (HuggingFace / OpenRouter)


import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable
from loguru import logger

from config import Config
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.vlm_handler import VLMHandler
from src.chunking.semantic_chunker import SemanticChunker, TextChunk
from src.embeddings.embedder import Embedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_handler import LLMHandler
from src.extraction.entity_extractor import EntityExtractor


class DocumentPipeline:
    """
    Main pipeline orchestrator.

    Always store in st.session_state so chunks survive Streamlit reruns:
        if "pipeline" not in st.session_state:
            st.session_state.pipeline = DocumentPipeline()
        pipeline = st.session_state.pipeline
    """

    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback or (lambda step, pct: None)

        # Lazy-loaded components — created only when first used
        self._loader    = None
        self._vlm       = None

        self._chunker   = None
        self._embedder  = None
        self._llm       = None
        self._extractor = None

        # State kept in memory
        # key = original filename (e.g. "report.pdf")
        self.indexed_documents: Dict[str, dict] = {}
        self.all_chunks:        Dict[str, list] = {}

        os.makedirs(Config.UPLOAD_DIR, exist_ok=True)

    #  Lazy Properties
    # Each component is created once and reused.

    @property
    def loader(self):
        if not self._loader:
            self._loader = DocumentLoader()
        return self._loader

    @property
    def vlm(self):
        if not self._vlm:
            try:
                self._vlm = VLMHandler()
            except Exception as e:
                logger.warning(f"VLM unavailable: {e}")
        return self._vlm

    @property
    def chunker(self):
        if not self._chunker:
            self._chunker = SemanticChunker()
        return self._chunker

    @property
    def embedder(self):
        if not self._embedder:
            self._embedder = Embedder()
        return self._embedder

    @property
    def llm(self):
        if not self._llm:
            self._llm = LLMHandler()
        return self._llm

    @property
    def extractor(self):
        if not self._extractor:
            self._extractor = EntityExtractor()
        return self._extractor


    # STAGE 1-4: Index a Document

    def index(self, file_path: str, original_filename: str = None) -> dict:
        """
        Full ingestion pipeline: Load -> VLM -> Chunk -> Embed -> Store.

        Args:
            file_path:          Path to the file on disk (may be a temp path).
            original_filename:  The user-facing filename (e.g. "report.pdf").
                                Always pass this from the UI so the collection
                                name matches what query() will look up later.
                                If not given, falls back to the file path name.

        Returns:
            Summary dict with stats about what was indexed.
        """
        start = time.time()

        # Use the original filename as the collection key, not the temp path name.
        # This is critical — the UI passes active_doc = original_filename,
        # so the collection name must match exactly.
        filename = original_filename if original_filename else Path(file_path).name
        logger.info(f"Indexing: {filename} (from path: {file_path})")

        # Stage 1: Load document
        self.progress_callback("Loading document...", 10)
        pages = self.loader.load(file_path)
        self._normalize_page_sources(pages, filename)
        logger.info(f"Loaded {len(pages)} pages from {filename}")

        visual_pages = [p for p in pages if p.content_type in ("image", "scanned")]
        table_pages  = [p for p in pages if p.content_type == "table"]

        # Stage 2: Process visual pages with VLM (charts, diagrams, scans)
        if visual_pages:
            self.progress_callback(
                f"Processing {len(visual_pages)} visual page(s) with VLM...", 30
            )
            pages = self._process_visual_pages(pages)

        # Stage 3: Chunk text semantically using LangChain splitter
        self.progress_callback("Chunking document...", 55)
        chunks = self.chunker.chunk(pages)
        logger.info(f"Created {len(chunks)} chunks for {filename}")

        # Store chunks in memory for BM25 retrieval
        self.all_chunks[filename] = chunks

        # Stage 4: Embed chunks and store in ChromaDB
        self.progress_callback("Embedding and indexing...", 75)
        num_indexed = self.embedder.index(chunks, collection_name=filename)

        elapsed = round(time.time() - start, 2)
        summary = {
            "filename":            filename,
            "total_pages":         len(pages),
            "visual_pages":        len(visual_pages),
            "table_pages":         len(table_pages),
            "total_chunks":        len(chunks),
            "indexed_chunks":      num_indexed,
            "processing_time_sec": elapsed,
        }
        self.indexed_documents[filename] = summary
        self.progress_callback("Done! Document is ready to query.", 100)
        logger.info(f"Indexing complete: {summary}")
        return summary

    def _normalize_page_sources(self, pages: list, filename: str) -> None:
        """Keep user-facing source metadata stable even when parsing temp files."""
        for page in pages:
            page.source_file = filename
            page.metadata["source"] = filename

    def _process_visual_pages(self, pages: list) -> list:
        """Run VLM or OCR on image/scanned pages, pass others through unchanged."""
        processed = []
        for page in pages:
            if page.content_type in ("image", "scanned") and page.image_bytes:
                if self.vlm:
                    page = self.vlm.process(page)
                elif self.ocr.available:
                    page.text = self.ocr.extract(page.image_bytes)
                    page.metadata["ocr_processed"] = True
            processed.append(page)
        return processed


    # STAGE 5-6: Query

    def query(
        self,
        question: str,
        collection_name: Optional[str] = None,
        mode: str = "qa",
    ) -> dict:
        """
        Answer a question using the full RAG pipeline.

        Args:
            question:        The user's question (string).
            collection_name: Filename of the document to search.
                             Pass st.session_state.active_doc from the UI.
            mode:            "qa" | "extract" | "summarize" | "anomaly"

        Returns:
            dict with answer, sources, entities, anomalies, retrieval stats.
        """
        # Check ChromaDB has at least one collection
        available = self.embedder.get_collection_names()
        if not available:
            return {
                "answer": (
                    "No documents have been indexed yet. "
                    "Please upload a document using the sidebar first."
                ),
                "sources": [], "entities": {}, "anomalies": [],
            }

        # Resolve which collection to search
        if not collection_name:
            # Try in-memory index first, then fall back to ChromaDB list
            collection_name = (
                list(self.indexed_documents.keys())[-1]
                if self.indexed_documents
                else None
            )

        if not collection_name:
            return {
                "answer": (
                    "Please select a document from the sidebar before asking questions."
                ),
                "sources": [], "entities": {}, "anomalies": [],
            }

        # Rebuild chunks from ChromaDB if they were lost (e.g. app restart,
        # Streamlit session expiry, or first query after re-opening the app).
        # This is what makes BM25 work even after a session gap.
        if not self.all_chunks.get(collection_name):
            logger.info(
                f"Chunks not in memory for '{collection_name}' — "
                "rebuilding from ChromaDB..."
            )
            rebuilt = self._rebuild_chunks_from_chroma(collection_name)
            if rebuilt:
                self.all_chunks[collection_name] = rebuilt
            else:
                logger.warning(
                    f"Could not rebuild chunks for '{collection_name}'. "
                    "The document may need to be re-indexed."
                )

        chunks = self.all_chunks.get(collection_name, [])

        # Stage 5: Hybrid Retrieval (BM25 + Dense + Reranker)
        retriever = HybridRetriever(embedder=self.embedder, chunks=chunks)
        retrieved = retriever.retrieve(
            query=question,
            collection_name=collection_name,
            top_k=Config.RERANKER_TOP_K,
        )

        if not retrieved:
            return {
                "answer": (
                    "I couldn't find relevant content in the document for this question.\n\n"
                    "Suggestions:\n"
                    "- Try rephrasing your question\n"
                    "- Ask about a specific section or topic in the document\n"
                    "- Make sure the document was indexed successfully (check sidebar)"
                ),
                "sources": [], "entities": {}, "anomalies": [],
            }

        # Stage 6: Generate answer with LLM
        llm_response = self.llm.answer(
            query=question,
            retrieved_chunks=retrieved,
            mode=mode,
        )

        # Optional extras for extract and anomaly modes
        entities  = {}
        anomalies = []
        if mode in ("extract", "anomaly"):
            entities = self.extractor.extract_from_chunks(retrieved)
        if mode == "anomaly":
            anomalies = self.extractor.detect_anomalies(retrieved)

        return {
            "answer":              llm_response["answer"],
            "sources":             llm_response["sources"],
            "model":               llm_response.get("model"),
            "quality":             llm_response.get("quality", {}),
            "entities":            entities,
            "anomalies":           anomalies,
            "context_chunks_used": llm_response["context_used"],
            "retrieval": {
                "total_candidates": len(retrieved),
                "bm25_count":  sum(
                    1 for r in retrieved if r.get("retrieval_method") == "bm25"
                ),
                "dense_count": sum(
                    1 for r in retrieved if r.get("retrieval_method") == "dense"
                ),
            },
        }

    def _rebuild_chunks_from_chroma(self, collection_name: str) -> List[TextChunk]:
        """
        Rebuild TextChunk objects from ChromaDB storage.

        Called when self.all_chunks is empty for a document that exists in
        ChromaDB. This happens after Streamlit reruns, app restarts, or when
        the pipeline object is recreated.

        ChromaDB stores both the text and the metadata for every chunk,
        so we can reconstruct full TextChunk objects from it. These are
        enough for BM25 keyword search.
        """
        try:
            clean_name = self.embedder._clean_collection_name(collection_name)
            collection = self.embedder.chroma_client.get_collection(
                name=clean_name,
                embedding_function=None,
            )

            count = collection.count()
            if count == 0:
                logger.warning(f"Collection '{clean_name}' exists but is empty")
                return []

            # Fetch all stored texts and their metadata from ChromaDB
            all_data = collection.get(
                include=["documents", "metadatas"],
                limit=count,
            )

            chunks = []
            for i, (text, meta) in enumerate(
                zip(all_data["documents"], all_data["metadatas"])
            ):
                chunk = TextChunk(
                    text=text,
                    source_file=meta.get("source_file", collection_name),
                    page_number=int(meta.get("page_number", 0)),
                    section_title=meta.get("section_title", ""),
                    chunk_index=int(meta.get("chunk_index", i)),
                    content_type=meta.get("content_type", "text"),
                    metadata=meta,
                )
                chunks.append(chunk)

            logger.info(
                f"Rebuilt {len(chunks)} chunks from ChromaDB for '{collection_name}'"
            )
            return chunks

        except Exception as e:
            logger.warning(f"Could not rebuild chunks from ChromaDB: {e}")
            return []


    # Utility Methods


    def reset_conversation(self):
        """Clear LLM conversation history. Call when switching documents."""
        if self._llm:
            self._llm.reset_conversation()

    def get_document_list(self) -> List[str]:
        """Return list of indexed document filenames."""
        return list(self.indexed_documents.keys())

    def remove_document(self, filename: str):
        """Remove a document from both ChromaDB and in-memory state."""
        self.embedder.delete_collection(filename)
        self.indexed_documents.pop(filename, None)
        self.all_chunks.pop(filename, None)
        logger.info(f"Removed document: {filename}")
