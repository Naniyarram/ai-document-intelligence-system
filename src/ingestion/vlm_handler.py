# src/ingestion/vlm_handler.py

# STAGE 2 — Vision Language Model (VLM) Handler


import base64
from loguru import logger
from config import Config
from src.ingestion.document_loader import DocumentPage


PROMPTS = {
    "image": (
        "This is an image from a document. Extract ALL information visible in it.\n"
        "- Chart or graph: describe title, axes, all data values, and key trends.\n"
        "- Diagram or flowchart: list all nodes/boxes and their connections.\n"
        "- Table: extract all rows and columns, use ' | ' as column separator.\n"
        "- Screenshot: extract all visible text and UI labels.\n"
        "Format as structured plain text suitable for search indexing."
    ),
    "scanned": (
        "This is a scanned document page. Extract ALL text exactly as it appears.\n"
        "Preserve the structure: paragraphs, numbered lists, bullet points, headings.\n"
        "For tables, format each row on one line with columns separated by ' | '.\n"
        "Do not summarise — extract every word you can see."
    ),
    "table": (
        "This image contains a table. Extract it completely.\n"
        "Format: put one row per line, separate columns with ' | '.\n"
        "Include the header row first, then all data rows. Skip nothing."
    ),
}


class VLMHandler:
    """
    Sends document images to a Vision Language Model for text extraction.

    Supports HuggingFace Inference Providers and OpenRouter.
    The correct API URL and key are loaded from config automatically.

    Usage:
        vlm = VLMHandler()
        page = vlm.process(page)   # page.image_bytes → page.text
    """

    def __init__(self):
        from openai import OpenAI
        import httpx

        # Use the same helper methods as llm_handler.py
        # These auto-detect whether to use HuggingFace or OpenRouter
        api_key  = Config.get_api_key()
        base_url = Config.get_base_url()
        backend  = Config.get_backend_name()

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(
                trust_env=False,
                timeout=httpx.Timeout(90.0, connect=20.0),
            ),
        )
        self.model  = Config.get_vlm_model()

        logger.info(f"VLM ready | backend={backend} | model={self.model}")

    def process(self, page: DocumentPage) -> DocumentPage:
        """
        Extract text from a single page that contains an image.

        Fills in page.text with the VLM's output.
        Returns the same page object (modified in place).
        If extraction fails, page.text gets an error placeholder so the
        rest of the pipeline can continue without crashing.
        """
        if page.image_bytes is None:
            logger.warning("VLM.process called on page with no image — skipping")
            return page

        # Pick the most appropriate prompt for this content type
        prompt = PROMPTS.get(page.content_type, PROMPTS["image"])

        try:
            extracted = self._call_vlm(
                image_bytes=page.image_bytes,
                image_format=page.image_format,
                prompt=prompt,
            )
            page.text                      = extracted
            page.metadata["vlm_processed"] = True
            page.metadata["vlm_model"]     = self.model

            logger.info(
                f"VLM: extracted {len(extracted)} chars from "
                f"'{page.content_type}' page {page.page_number} "
                f"in {page.source_file}"
            )

        except Exception as e:
            logger.error(f"VLM failed on page {page.page_number}: {e}")

            # Use a placeholder so the chunk is still indexed (not lost)
            page.text                   = f"[Visual content — VLM extraction failed: {str(e)}]"
            page.metadata["vlm_failed"] = True

        return page

    def process_batch(self, pages: list) -> list:
        """
        Process a list of pages, running VLM only on visual ones.
        Text-only pages are passed through unchanged.
        """
        visual_types = {"image", "scanned", "table"}
        return [
            self.process(p) if (p.content_type in visual_types and p.image_bytes)
            else p
            for p in pages
        ]

    def _call_vlm(self, image_bytes: bytes, image_format: str, prompt: str) -> str:
        """
        Make the API call to the VLM with the image encoded as base64.

        Both HuggingFace and OpenRouter accept images this way because
        they both implement the OpenAI chat completions format.
        """
        b64_image  = base64.b64encode(image_bytes).decode("utf-8")
        media_type = f"image/{image_format}"

        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        last_error = None
        for model_name in self._candidate_models():
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=message,
                    max_tokens=1000,
                )
                if model_name != self.model:
                    logger.warning(
                        f"Configured VLM '{self.model}' unavailable; switched to '{model_name}'"
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
                logger.warning(f"VLM model '{model_name}' unavailable: {exc}")

        raise last_error

    def _candidate_models(self) -> list[str]:
        """Return the configured VLM plus backend-specific fallbacks."""
        ordered = [self.model] + Config.get_vlm_fallback_models()
        unique = []
        for model_name in ordered:
            if model_name and model_name not in unique:
                unique.append(model_name)
        return unique

    @staticmethod
    def _is_model_unavailable_error(error: Exception) -> bool:
        """Identify provider errors that mean the model ID is unsupported or rate limited."""
        message = str(error).lower()
        return any(
            clue in message
            for clue in ("404", "429", "model", "not found", "does not exist", "unavailable", "rate", "upstream")
        )

