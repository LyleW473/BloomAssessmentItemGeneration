"""
Document parsing utilities for course-content extraction.
- Exposes `ParserService`, a thin wrapper around LlamaParse that turns uploaded
  files (PDF / PPT / DOCX / MD) into extracted markdown text.
"""
from .service import ParserService

__all__ = ["ParserService"]
