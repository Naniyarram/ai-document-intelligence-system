
# src/chunking/semantic_chunker.py

# STAGE 3 — LangChain-based Chunking


from dataclasses import dataclass, field
from typing import List
from loguru import logger
from config import Config


@dataclass
class TextChunk:
    """
    One retrievable unit of text from a document.
    This is what gets embedded and stored in the vector database.
    """
    text: str                   # The actual text content
    source_file: str            # Which file this came from
    page_number: int            # Which page/section
    section_title: str = ""     # Section heading (if any)
    chunk_index: int = 0        # Position within the document
    content_type: str = "text"  # "text", "table", "spreadsheet_row"

    # Metadata stored alongside the vector in ChromaDB
    metadata: dict = field(default_factory=dict)

    def to_metadata_dict(self) -> dict:
        """
        Convert to a flat dict for ChromaDB metadata storage.
        ChromaDB only accepts simple types (str, int, float, bool).
        """
        base = {
            "source_file":   self.source_file,
            "page_number":   self.page_number,
            "section_title": self.section_title,
            "chunk_index":   self.chunk_index,
            "content_type":  self.content_type,
        }
        # Merge in extra metadata, but only keep ChromaDB-safe types
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                base[k] = v
            else:
                base[k] = str(v)  # convert everything else to string
        return base


class SemanticChunker:
    """
    Splits DocumentPage objects into TextChunk objects using LangChain.

    Strategy:
      - Regular text  → LangChain RecursiveCharacterTextSplitter
                        (respects paragraphs → sentences → words)
      - Tables        → kept as ONE chunk (never split a table)
      - Spreadsheets  → kept as ONE chunk per batch of rows

    Why RecursiveCharacterTextSplitter is great:
      It tries to split on natural boundaries in order:
        paragraph breaks → line breaks → sentences → words
      Only falls back to the next level if the chunk is still too big.
      This produces clean, readable chunks almost every time.

    Usage:
        chunker = SemanticChunker()
        chunks = chunker.chunk(pages)   # List[DocumentPage] → List[TextChunk]
    """

    def __init__(self):
        self.chunk_size    = Config.CHUNK_SIZE     # ~400 tokens
        self.chunk_overlap = Config.CHUNK_OVERLAP  # ~60 tokens overlap

        # One token ≈ 4 characters — so we convert token targets to char targets
        # for LangChain (which works in characters, not tokens)
        self._char_size    = self.chunk_size    * 4  # 400 tokens → ~1600 chars
        self._char_overlap = self.chunk_overlap * 4  # 60 tokens  → ~240 chars

        self._splitter = self._build_splitter()
        logger.info(
            f"LangChain chunker ready "
            f"(chunk_size={self._char_size} chars, "
            f"overlap={self._char_overlap} chars)"
        )

    def _build_splitter(self):
        """
        Build the LangChain RecursiveCharacterTextSplitter.

        The separators list is tried IN ORDER — it only moves to the
        next separator if the current chunk is still too large.

        This means:
          - It PREFERS to split on paragraph breaks (\n\n)
          - Falls back to line breaks (\n)
          - Falls back to sentence endings (". ")
          - Falls back to spaces
          - Only splits mid-word as an absolute last resort
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=self._char_size,
            chunk_overlap=self._char_overlap,
            length_function=len,       # measure in characters
            separators=[
                "\n\n",   # paragraph break (most preferred)
                "\n",     # line break
                ". ",     # sentence ending
                "! ",     # exclamation sentence
                "? ",     # question sentence
                "; ",     # clause break
                ", ",     # comma clause
                " ",      # word boundary
                "",       # character level (last resort)
            ],
            is_separator_regex=False,
        )

    def chunk(self, pages: list) -> List[TextChunk]:
        """
        Main entry point.
        Takes all pages from a document, returns a flat list of TextChunks.

        Args:
            pages: List[DocumentPage] from the ingestion stage

        Returns:
            List[TextChunk] ready for embedding
        """
        all_chunks = []
        global_chunk_index = 0

        for page in pages:
            # Skip pages with no text content
            if not page.text or not page.text.strip():
                continue

            # Tables & spreadsheets: NEVER split them
            # A table chunk that gets cut in half is useless.
            # Keep the entire table as a single retrievable unit.
            if page.content_type in ("table", "spreadsheet_row"):
                chunk = TextChunk(
                    text=page.text.strip(),
                    source_file=page.source_file,
                    page_number=page.page_number,
                    section_title=page.section_title,
                    chunk_index=global_chunk_index,
                    content_type=page.content_type,
                    metadata=page.metadata,
                )
                all_chunks.append(chunk)
                global_chunk_index += 1
                continue

            #  Regular text: use LangChain splitter
            page_chunks = self._split_with_langchain(
                text=page.text,
                source_file=page.source_file,
                page_number=page.page_number,
                section_title=page.section_title,
                content_type=page.content_type,
                metadata=page.metadata,
                start_index=global_chunk_index,
            )
            all_chunks.extend(page_chunks)
            global_chunk_index += len(page_chunks)

        logger.info(
            f"Chunking complete: {len(all_chunks)} chunks "
            f"from {len(pages)} pages"
        )
        return all_chunks

    def _split_with_langchain(
        self,
        text: str,
        source_file: str,
        page_number: int,
        section_title: str,
        content_type: str,
        metadata: dict,
        start_index: int,
    ) -> List[TextChunk]:
        """
        Use LangChain's splitter on a single page's text.
        Converts the resulting string splits back into TextChunk objects.
        Prepends the section_title to preserve semantic context.
        """
        # LangChain returns plain strings
        raw_chunks: List[str] = self._splitter.split_text(text)

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue  # skip empty splits

            # Embed the section title directly into the chunk text so the vector DB
            # and the LLM both see the explicit context.
            if section_title and section_title not in chunk_text:
                # Format: [SECTION TITLE] \n\n Body text...
                final_text = f"[{section_title}]\n\n{chunk_text}"
            else:
                final_text = chunk_text

            chunks.append(TextChunk(
                text=final_text,
                source_file=source_file,
                page_number=page_number,
                section_title=section_title,
                chunk_index=start_index + i,
                content_type=content_type,
                metadata=metadata,
            ))

        return chunks
