"""
Markdown-splitting helpers used by `ParserService` to turn parsed documents into chunks.
- `split_markdown_by_headers_and_pages`: header/page-aware first-pass split.
- `process_pdf_docs_markdown`: hierarchical split (headers then size-based) for PDF/DOCX/MD.
- `process_ppt_markdown`: slide-based split for PowerPoint exports.

Note: the extraction pipeline only uses the full document text returned by the parser;
these chunkers are kept so `ParserService` can also report a chunk count per file.
"""
import re
from typing import Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.schema import Document


def split_markdown_by_headers_and_pages(doc: Document) -> Tuple[List[str], List[Dict]]:
    """
    Phase 1: Split markdown file based on headers and pages.
    - Header hierarchy information is added to the content itself for better context.
    
    Args:
        doc (Document): The document to split.
    Returns:
        Tuple[List[str], List[Dict]]: A tuple containing the list of chunk texts and their corresponding metadata.
    """
    docs_text = doc.text
    file_path = doc.metadata['file_name']
    lines = docs_text.split('\n')
    headers_to_split_on = [
        ("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"),
        ("####", "Header 4"), ("#####", "Header 5"), ("######", "Header 6"),
    ]
    header_map = {h[0]: h[1] for h in headers_to_split_on}

    chunks = []
    current_content_lines = []
    current_headers = {}
    start_page_for_chunk = 1
    current_page = 1

    def flush_chunk():
        """
        Flush the current content lines into a chunk, if any.

        - The chunk's content is prefixed with the current header hierarchy.
        - Metadata includes the current headers and page range.
        """
        if not current_content_lines:
            return

        header_prefix = ""
        for _, header_key in headers_to_split_on:
            if header_key in current_headers:
                header_prefix += f"{current_headers[header_key]} > "
        header_prefix = header_prefix[:-3] if header_prefix else ""

        raw_content = "".join(current_content_lines).strip()
        if not raw_content:
            return

        final_content = f"{header_prefix}\n\n{raw_content}" if header_prefix else raw_content
        metadata = current_headers.copy()
        metadata['source_file'] = file_path
        metadata['page_start'] = start_page_for_chunk
        metadata['page_end'] = current_page
        chunks.append({"content": final_content, "metadata": metadata})

    for line in lines:
        # Page delimiter
        if line.strip() == '---':
            current_page += 1
            continue

        match = re.match(r'^(#+)\s+(.*)', line.strip())
        header_hashes = match.group(1) if match else None

        if match and header_hashes in header_map:
            flush_chunk()
            current_content_lines = []
            start_page_for_chunk = current_page

            header_level = len(header_hashes)
            header_key_name = header_map[header_hashes]
            header_text = match.group(2).strip()

            # Clear headers below the current level
            keys_to_clear = [h_value for h_key, h_value in header_map.items() if len(h_key) > header_level and h_value in current_headers]
            for key in keys_to_clear:
                del current_headers[key]

            current_headers[header_key_name] = header_text

        current_content_lines.append(line)

    flush_chunk()
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    return texts, metadatas

def process_pdf_docs_markdown(doc: Document, chunk_size: int = 500, chunk_overlap: int = 50) -> Tuple[List[str], List[Dict]]:
    """
    Phase 2: Hierarchical splitting of PDF/DOCX/MD content.
    - First, split by headers and pages.
    - Then, for chunks exceeding `chunk_size`, perform a secondary split based on size and overlap.

    Args:
        doc (Document): The document to split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.
    Returns:
        Tuple[List[str], List[Dict]]: A tuple containing the list of chunk texts and their corresponding metadata.
    """
    initial_texts, initial_metadatas = split_markdown_by_headers_and_pages(doc)
    secondary_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    final_texts, final_metadatas = [], []

    for text, metadata in zip(initial_texts, initial_metadatas):
        if len(text) <= chunk_size:
            final_texts.append(text)
            final_metadatas.append(metadata)
        else:
            sub_chunks = secondary_splitter.split_text(text)
            for sub_chunk_text in sub_chunks:
                final_texts.append(sub_chunk_text)
                final_metadatas.append(metadata.copy())

    print(f"Phase 1 splitting created {len(initial_texts)} chunks.")
    print(f"Hierarchical splitting created {len(final_texts)} chunks in total.")
    return final_texts, final_metadatas

def process_ppt_markdown(doc: Document) -> Tuple[List[str], List[Dict]]:
    """
    Process a PowerPoint markdown document by splitting it into individual slides.

    Args:
        doc (Document): The PowerPoint markdown document to process.

    Returns:
        Tuple[List[str], List[Dict]]: A tuple containing the list of slide texts and their corresponding metadata.
    """
    ppt_markdown = doc.text
    file_path = doc.metadata['file_name']
    slides = re.split(r'\n---\n', ppt_markdown.strip())

    final_texts = []
    final_metadatas = []
    for i, slide_content in enumerate(slides):
        cleaned_content = slide_content.strip()
        if not cleaned_content:
            continue

        page_number = i + 1

        metadata = {
            "source_file": file_path,
            "page": page_number
        }

        final_texts.append(cleaned_content)
        final_metadatas.append(metadata)

    return final_texts, final_metadatas
