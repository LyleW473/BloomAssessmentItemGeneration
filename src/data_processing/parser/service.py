import os
import logging
from llama_index.core.schema import Document

from src.data_processing.parser.utils import process_pdf_docs_markdown, process_ppt_markdown

logger = logging.getLogger(__name__)


class ParserService:
    def __init__(self, llama_cloud_api_key: str):
        # Lazy import so the heavy LlamaParse dependency is only required when a file
        # actually needs parsing (e.g. plain .txt inputs don't touch this code path).
        from llama_parse import LlamaParse

        # Initialize LlamaParse with default parameters
        self.parser = LlamaParse(
            api_key=llama_cloud_api_key,  # Your Llama Cloud API key
            result_type="markdown",   # Output format: "markdown" or "text"
            num_workers=2,            # Number of parallel workers
            verbose=True,             # Print detailed logs
            split_by_page=False       # Whether to split results by page
        )

    def parse_material(self, file_path: str, subject: str = "default", version: int = 0) -> dict:
        """
        Parse a single file (given by its filesystem path) and return chunk information.

        Synchronous on purpose: LlamaParse's `load_data` already manages its own event loop
        internally, so it must be called from plain synchronous code. Wrapping it in an extra
        `asyncio.run(...)` per file binds LlamaParse's cached HTTP client to a loop that then
        closes, which breaks multi-call/concurrent parsing (e.g. PPTX) with "Event loop is closed".

        Supported file types:
        - PDF (.pdf)
        - PowerPoint (.ppt, .pptx)
        - Word (.docx)
        - Markdown (.md)

        Args:
            file_path (str): Path to the file to parse.
            subject (str): Optional subject/course name for logging.
            version (int): Optional version number for logging.
        Returns:
            dict: A dictionary containing the parsing result, including status, subject,
                  filename, type, number of chunks, and extracted texts.
        """
        # Computed before the try so the `except` branch can always reference it.
        filename = os.path.basename(file_path)
        try:
            file_ext = os.path.splitext(file_path)[-1].lower()
            extra_info = {"file_name": filename}

            # Process different file types. LlamaParse accepts the real path directly,
            # so there is no need to copy bytes to /tmp first.
            if file_ext in [".pdf", ".docx", ".ppt", ".pptx"]:
                docs = self.parser.load_data(file_path, extra_info=extra_info)

                if not docs:
                    raise ValueError(f"No content extracted from {filename}")

                doc = docs[0]
                result_txt = doc.text
                # Route to specialized processors based on type
                if file_ext in [".ppt", ".pptx"]:
                    texts, metadatas = process_ppt_markdown(doc)
                else:
                    texts, metadatas = process_pdf_docs_markdown(doc)

            elif file_ext == ".md":
                # For markdown, read the file directly and wrap into a Document object.
                with open(file_path, "r", encoding="utf-8") as f_md:
                    result_txt = f_md.read()

                doc = Document(text=result_txt, metadata=extra_info)
                texts, metadatas = process_pdf_docs_markdown(doc)

            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            logger.info(
                f"Parsed file {filename} (type={file_ext}) "
                f"for course {subject}, chunks={len(texts)}"
            )

            return {
                "status": "success",
                "subject": subject,
                "filename": filename,
                "type": 'Material',
                "chunks": len(texts),
                "texts": result_txt,
            }

        except Exception as e:
            logger.error(
                f"Error parsing file {filename} for {subject}: {e}",
                exc_info=True
            )
            return {
                "status": "error",
                "subject": subject,
                "filename": filename,
                "error": str(e),
                "type": 'Material',
            }
