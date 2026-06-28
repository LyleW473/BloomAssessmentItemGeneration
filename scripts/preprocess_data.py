"""
End-to-end preprocessing entry point for the assessment question-generation framework.

It runs two sequential phases:

  Phase 1: Extract + classify
    For every resolved file: extract text (.txt read directly; .pdf/.ppt/.pptx/.docx/.md
    parsed via LlamaParse; .mp4 skipped), then classify it as
    course_concept / course_adjacent / administrative.

  Phase 2: Compress
    High-fidelity compression of each extracted text into `compressed_text`, plus a
    per-module metadata file recording compression ratios.

All provided files are grouped into a single module JSON at:
- {BASE_EXTRACTED_QUESTIONS_DATA_DIR}/contents/{name}.json

Output schema (kept identical to the original pipeline so downstream code is unchanged):
    {
      "sorted_weeks_contents": ["<week_dir>", ...],
      "extracted_texts": {
        "<file_name>": {
          "file_path": "<path>",
          "week_dir": "<parent folder name or 'n/a'>",
          "text": "<extracted text>",
          "classification": {"category": "...", "reason": "..."},
          "compressed_text": "<compressed text>"
        },
        ...
      }
    }

Usage examples:
    # A single file
    python scripts/preprocess_data.py path/to/lecture.pdf

    # Several files into one named module
    python scripts/preprocess_data.py a.pdf b.pptx notes.txt --name intro_ml

    # A whole folder (every file inside, recursively)
    python scripts/preprocess_data.py ./my_course/ --name intro_ml

    # Force everything to be usable downstream, even if not classified course_concept
    python scripts/preprocess_data.py lecture.pdf --force-concept

    # Choose models per run (slugs go to the OpenAI-compatible endpoint set by OPENAI_BASE_URL).
    # With OpenRouter (OPENAI_BASE_URL=https://openrouter.ai/api/v1), use provider-namespaced slugs:
    python scripts/preprocess_data.py lecture.pdf \\
        --classification-model openai/gpt-4o \\
        --compression-model google/gemini-3.5-flash
"""
import set_path
import argparse
import json
import os
import time

from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI, APIStatusError

from src.globals import BASE_EXTRACTED_QUESTIONS_DATA_DIR
from src.data_processing.parser.service import ParserService
from src.data_processing.synthetic.utils import get_all_paths_in_dir, sort_contents_by_week_number
from src.llm_response_generation.functions import generate_llm_response, extract_json_from_text
from src.data_processing.synthetic.bloom_system_instructions import (
    COURSE_CONTENT_CLASSIFICATION_SYSTEM_INSTRUCTION,
    COMPRESSION_INSTRUCTION,
)

# Default model slugs for each LLM step. These are passed to an OpenAI-compatible endpoint,
# so when using OpenRouter they must be provider-namespaced (e.g. "openai/gpt-4o").
# Override per-run with --classification-model / --compression-model.
DEFAULT_CLASSIFICATION_MODEL = "openai/gpt-4o"
DEFAULT_COMPRESSION_MODEL = "google/gemini-3.5-flash"

# File handling.
VALID_CATEGORIES = [
    "course_concept", 
    "course_adjacent", 
    "administrative"
]
SKIPPED_EXTENSIONS = {".mp4"} # TODO: Add video parsing for transcripts
PLAIN_TEXT_EXTENSIONS = {".txt"}
PARSEABLE_EXTENSIONS = {".pdf", ".ppt", ".pptx", ".docx", ".md"}


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------
def resolve_input_files(paths: List[str]) -> List[str]:
    """
    Expand the user-provided paths into a flat, de-duplicated list of files.
    - A file path is taken as-is.
    - A directory is expanded recursively into all the files it contains.

    Args:
        paths (List[str]): List of file and/or directory paths provided by the user.
    Returns:
        List[str]: A flat list of resolved file paths.
    """
    resolved: List[str] = []
    seen = set()
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if os.path.isfile(path):
            candidate_files = [path]
        else:
            candidate_files = sorted(get_all_paths_in_dir(path))

        for f in candidate_files:
            abs_f = os.path.abspath(f)
            if abs_f not in seen:
                seen.add(abs_f)
                resolved.append(f)

    if not resolved:
        raise ValueError("No files found in the provided path(s).")
    return resolved

def determine_module_name(paths: List[str], name_arg: Optional[str]) -> str:
    """
    Decide the output module name (-> contents/{name}.json).
    - Explicit --name always wins.
    - A single file defaults to its filename stem.
    - A single directory defaults to the directory's basename.
    - Multiple loose paths require an explicit --name.

    Args:
        paths (List[str]): List of file and/or directory paths provided by the user.
        name_arg (Optional[str]): The --name argument provided by the user, if any.
    Returns:
        str: The determined module name.
    Raises:
        ValueError: If multiple paths are provided without an explicit --name.
    """
    if name_arg:
        return name_arg

    if len(paths) == 1:
        only = paths[0].rstrip(os.sep)
        if os.path.isdir(only):
            return os.path.basename(only)
        return os.path.splitext(os.path.basename(only))[0]

    raise ValueError(
        "Multiple paths were provided; please pass --name to choose the output module name."
    )

def week_dir_for(file_path: str) -> str:
    """
    Best-effort 'week' grouping for a file: the name of its immediate parent folder.

    Args:
        file_path (str): The path to the file.
    Returns:
        str: The name of the immediate parent folder, or "n/a" if not available.
    """
    parent = os.path.basename(os.path.dirname(os.path.abspath(file_path)))
    return parent if parent else "n/a"

# ---------------------------------------------------------------------------
# Phase 1 helpers: extraction + classification
# ---------------------------------------------------------------------------
def extract_text_from_file(
    content_path: str,
    get_parser_service,
) -> Optional[str]:
    """
    Extract raw text from a single file.
    - Returns the extracted text, or None if the file should be skipped.
    - `get_parser_service` is a zero-arg callable that lazily builds/returns a ParserService
      (so a LlamaParse key is only required when a parseable file is actually encountered).
    
    Args:
        content_path (str): The path to the file to extract text from.
        get_parser_service (Callable[[], ParserService]): A callable that returns a ParserService instance.
    Returns:
        Optional[str]: The extracted text, or None if the file should be skipped.
    """
    content_file_name = os.path.basename(content_path)
    ext = os.path.splitext(content_file_name)[-1].lower()

    if ext in SKIPPED_EXTENSIONS:
        print(f"Skipping unsupported file type '{ext}': {content_file_name}")
        return None

    if ext in PLAIN_TEXT_EXTENSIONS:
        with open(content_path, "r", encoding="utf-8") as f_txt:
            return f_txt.read()

    if ext in PARSEABLE_EXTENSIONS:
        # LlamaParse accepts the real path directly; call the (synchronous) parser on it.
        result: Dict[str, Any] = get_parser_service().parse_material(
            file_path=content_path, subject=content_file_name
        )

        if result["status"] == "error":
            print(f"Error parsing file {content_file_name}: {result['error']}")
            return None

        return result["texts"]

    print(f"Skipping unsupported file type '{ext}': {content_file_name}")
    return None

def classify_text(client: OpenAI, file_name: str, file_text: str, model_name: str) -> Dict[str, str]:
    """
    Classify extracted text into course_concept / course_adjacent / administrative.

    Args:
        client (OpenAI): The OpenAI client instance.
        file_name (str): The name of the file being classified.
        file_text (str): The text content of the file.
        model_name (str): The model slug to use for classification.
    Returns:
        Dict[str, str]: The classification result as a dictionary.
    """
    classification_input_json = {"file_name": file_name, "file_text": file_text}
    classification_response_text: str = generate_llm_response(
        client=client,
        model_name=model_name,
        messages=[
            {"role": "system", "content": COURSE_CONTENT_CLASSIFICATION_SYSTEM_INSTRUCTION},
            {"role": "user", "content": "Classify the following course content into one of the specified categories.\n\n" + json.dumps(classification_input_json)},
        ],
    )
    classification_json = extract_json_from_text(classification_response_text)

    assert classification_json is not None and isinstance(classification_json, dict), "Course content classification JSON is not a dict"
    assert "category" in classification_json, "Course content classification JSON missing 'category' field"
    assert classification_json["category"] in VALID_CATEGORIES, "Invalid classification value"
    return classification_json

def unique_key(file_name: str, existing: Dict[str, Any], file_path: str) -> str:
    """
    Pick a dict key for a file. Uses the basename, but disambiguates with the parent
    folder when two different files share a basename (the schema is keyed by file name).

    Args:
        file_name (str): The base name of the file.
        existing (Dict[str, Any]): The existing dictionary of extracted texts.
        file_path (str): The full path to the file.
    Returns:
        str: A unique key for the file, potentially including the parent folder.
    """
    if file_name not in existing:
        return file_name
    parent = os.path.basename(os.path.dirname(os.path.abspath(file_path)))
    candidate = f"{parent}/{file_name}" if parent else file_name
    suffix = 1
    while candidate in existing:
        suffix += 1
        candidate = f"{parent}/{file_name}#{suffix}"
    return candidate

# ---------------------------------------------------------------------------
# Phase 2 helpers: compression
# ---------------------------------------------------------------------------
def compress_text(
        client: OpenAI,
        file_name: str,
        file_text: str,
        model_name: str,
        max_retries: int = 10,
        base_delay: float = 1.0
    ) -> str:
    """
    High-fidelity compression of extracted text, with exponential backoff on failures.

    - Compresses the extracted text into a smaller representation while preserving essential information
      using an LLM.

    Args:
        client (OpenAI): The OpenAI client instance.
        file_name (str): The name of the file being compressed.
        file_text (str): The text content of the file to compress.
        model_name (str): The model slug to use for compression.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 10.
        base_delay (float, optional): Base delay in seconds for exponential backoff. Defaults to 1.0.
    Returns:
        str: The compressed text.
    """
    compressed_text_input_json = {"file_name": file_name, "file_text": file_text}
    compressed_text_json = None

    for attempt in range(1, max_retries + 1):
        try:
            compressed_text_response_text: str = generate_llm_response(
                client=client,
                model_name=model_name,
                messages=[
                    {"role": "system", "content": COMPRESSION_INSTRUCTION},
                    {"role": "user", "content": json.dumps(compressed_text_input_json)},
                ],
            )
            compressed_text_json = extract_json_from_text(compressed_text_response_text)

            if compressed_text_json and isinstance(compressed_text_json, dict) and "compressed_text" in compressed_text_json:
                return compressed_text_json["compressed_text"]

            raise ValueError("Invalid JSON payload or missing compressed_text")

        except Exception as e:
            # Fail fast on permanent errors (bad model slug, auth/permission) instead of
            # burning all retries on something that cannot succeed.
            status_code = getattr(e, "status_code", None)
            if isinstance(e, APIStatusError) and status_code in (400, 401, 403, 404):
                raise
            wait = min(base_delay * (2 ** (attempt - 1)), 30)
            print(f"Attempt {attempt}/{max_retries} failed: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed to get valid compressed_text after {max_retries} attempts for {file_name}")

def write_module_metadata(
        json_path: str, 
        extracted_texts: Dict[str, Dict[str, Any]]
    ) -> None:
    """
    Write a per-module metadata JSON recording raw/compressed lengths and compression ratios.
    
    Args:
        json_path (str): The path to the module JSON file.
        extracted_texts (Dict[str, Dict[str, Any]]): The dictionary of extracted texts
                                                     containing raw and compressed text 
                                                     for each file.
    Returns:
        None
    """
    per_file_metadata: Dict[str, Any] = {}
    course_concept_stats = {"count": 0, "raw_total": 0, "compressed_total": 0}
    non_course_concept_stats = {"count": 0, "raw_total": 0, "compressed_total": 0}

    for file_name, content_dict in extracted_texts.items():
        if "compressed_text" not in content_dict:
            continue
        raw_len = len(content_dict.get("text", ""))
        compressed_len = len(content_dict.get("compressed_text", ""))
        pct_decrease = (float(raw_len - compressed_len) / raw_len * 100.0) if raw_len > 0 else 0.0

        category = content_dict.get("classification", {}).get("category", "unknown")
        per_file_metadata[file_name] = {
            "raw_length": raw_len,
            "compressed_length": compressed_len,
            "percentage_decrease": round(pct_decrease, 4),
            "classification_category": category,
        }

        bucket = course_concept_stats if category == "course_concept" else non_course_concept_stats
        bucket["raw_total"] += raw_len
        bucket["compressed_total"] += compressed_len
        bucket["count"] += 1

    overall_raw_total = sum(item["raw_length"] for item in per_file_metadata.values())
    overall_compressed_total = sum(item["compressed_length"] for item in per_file_metadata.values())
    overall_pct = (float(overall_raw_total - overall_compressed_total) / overall_raw_total * 100.0) if overall_raw_total > 0 else 0.0

    def pct(stats: Dict[str, Any]) -> float:
        """
        Calculate the percentage decrease for a given stats dictionary.

        Args:
            stats (Dict[str, Any]): A dictionary containing 'raw_total' and 'compressed_total'.
        Returns:
            float: The percentage decrease, or 0.0 if 'raw_total' is zero.
        """
        return (float(stats["raw_total"] - stats["compressed_total"]) / stats["raw_total"] * 100.0) if stats["raw_total"] > 0 else 0.0

    metadata_dir = os.path.join(BASE_EXTRACTED_QUESTIONS_DATA_DIR, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_path = os.path.join(metadata_dir, os.path.basename(json_path).replace(".json", "_metadata.json"))

    with open(metadata_path, "w", encoding="utf-8") as f_meta:
        json.dump({
            "course_file": os.path.basename(json_path),
            "items": per_file_metadata,
            "overall_raw_total": overall_raw_total,
            "overall_compressed_total": overall_compressed_total,
            "overall_percentage_decrease": round(overall_pct, 4),
            "course_concept_stats": {
                "count": course_concept_stats["count"],
                "raw_total": course_concept_stats["raw_total"],
                "compressed_total": course_concept_stats["compressed_total"],
                "percentage_decrease": round(pct(course_concept_stats), 4),
            },
            "non_course_concept_stats": {
                "count": non_course_concept_stats["count"],
                "raw_total": non_course_concept_stats["raw_total"],
                "compressed_total": non_course_concept_stats["compressed_total"],
                "percentage_decrease": round(pct(non_course_concept_stats), 4),
            },
        }, f_meta, indent=4, ensure_ascii=False)
    print(f"Saved metadata to {metadata_path}")

def save_module(
        json_path: str, 
        module_data: Dict[str, Any]
    ) -> None:
    """
    Save the module data to a JSON file, creating directories as needed.

    Args:
        json_path (str): The path to the JSON file where the module data will be saved.
        module_data (Dict[str, Any]): The module data to save.
    Returns:
        None
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(module_data, f_json, indent=4, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the preprocessing script.

    Args:
        None
    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess course content (extract + classify + compress) into a generation-ready JSON module.",
    )
    parser.add_argument("paths", nargs="+", help="One or more files and/or directories to preprocess.")
    parser.add_argument("--name", default=None, help="Output module name -> contents/{name}.json. Defaults to the single file/dir name; required for multiple paths.")
    parser.add_argument("--force-concept", action="store_true", help="Force every file's category to 'course_concept' so it is always usable downstream.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess from scratch even if the module JSON already exists (default: merge new files in).")
    parser.add_argument("--classification-model", default=DEFAULT_CLASSIFICATION_MODEL, help=f"Model slug for the classification step (default: {DEFAULT_CLASSIFICATION_MODEL}). Use provider-namespaced slugs for OpenRouter, e.g. 'openai/gpt-4o'.")
    parser.add_argument("--compression-model", default=DEFAULT_COMPRESSION_MODEL, help=f"Model slug for the compression step (default: {DEFAULT_COMPRESSION_MODEL}). Use provider-namespaced slugs for OpenRouter, e.g. 'google/gemini-3.5-flash'.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load keys
    load_dotenv()
    OPENAI_LLM_API_KEY = os.getenv("OPENAI_LLM_API_KEY", None)
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", None)

    for key_name, key_val in [("OPENAI_LLM_API_KEY", OPENAI_LLM_API_KEY), ("OPENAI_BASE_URL", OPENAI_BASE_URL)]:
        if key_val is None:
            raise ValueError(f"{key_name} not found in environment variables")

    client = OpenAI(api_key=OPENAI_LLM_API_KEY, base_url=OPENAI_BASE_URL, timeout=60)

    # Lazily build the parser only when a parseable (non-txt) file is encountered, so that
    # text-only runs don't require a LLAMA_CLOUD_API_KEY.
    _parser_service: List[Optional[ParserService]] = [None]

    def get_parser_service() -> ParserService:
        """
        Lazily build and return a ParserService instance, using the LLAMA_CLOUD_API_KEY from
        environment variables. Raises an error if the key is not found and a parseable file is encountered.

        Args:
            None
        Returns:
            ParserService: An instance of the ParserService class.
        Raises:
            ValueError: If LLAMA_CLOUD_API_KEY is not found in environment variables when needed for parsing
                        e.g., for .pdf, .ppt, .pptx, .docx, or .md files.
        """
        if _parser_service[0] is None:
            if not LLAMA_CLOUD_API_KEY:
                raise ValueError("LLAMA_CLOUD_API_KEY not found in environment variables (required to parse non-.txt files)")
            _parser_service[0] = ParserService(llama_cloud_api_key=LLAMA_CLOUD_API_KEY)
        return _parser_service[0]

    # Resolve inputs and the output module path.
    input_files = resolve_input_files(args.paths)
    module_name = determine_module_name(args.paths, args.name)
    contents_dir = os.path.join(BASE_EXTRACTED_QUESTIONS_DATA_DIR, "contents")
    json_path = os.path.join(contents_dir, f"{module_name}.json")

    print(f"Module name: {module_name}")
    print(f"Output JSON: {json_path}")
    print(f"Resolved {len(input_files)} input file(s).")
    print(f"Classification model: {args.classification_model} | Compression model: {args.compression_model}")

    # Load existing module (merge mode) or start fresh (overwrite / new module).
    if os.path.exists(json_path) and not args.overwrite:
        with open(json_path, "r", encoding="utf-8") as f_json:
            module_data = json.load(f_json)
        extracted_texts: Dict[str, Dict[str, Any]] = module_data.get("extracted_texts", {})
        print(f"Merging into existing module ({len(extracted_texts)} file(s) already present).")
    else:
        module_data = {"sorted_weeks_contents": [], "extracted_texts": {}}
        extracted_texts = module_data["extracted_texts"]

    # Pin extracted_texts onto module_data so incremental save_module() calls always
    # capture the latest in-memory state (even if the loaded JSON lacked the key).
    module_data["extracted_texts"] = extracted_texts

    # Map already-processed source paths -> their dict key, so a merge run can skip
    # re-extraction (and, under --force-concept, override an existing entry in place).
    path_to_key = {
        os.path.abspath(entry["file_path"]): key
        for key, entry in extracted_texts.items()
        if isinstance(entry, dict) and "file_path" in entry
    }
    already_processed_paths = set(path_to_key.keys())

    non_concept_files: List[Tuple[str, str]] = [] # (file_name, category) for end-of-run summary
    failed_files: List[Tuple[str, str]] = [] # (file_name, error) for end-of-run summary

    # ---------------------------------------------------------------
    # Phase 1: extraction + classification
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 1: Extract + classify")
    print("=" * 80)

    def persist_phase1() -> None:
        """
        Refresh the derived week list and write the module to disk.
        
        Args:
            None
        Returns:
            None
        """
        week_dirs = sorted({entry["week_dir"] for entry in extracted_texts.values()})
        module_data["sorted_weeks_contents"] = sort_contents_by_week_number(week_dirs)
        save_module(json_path, module_data)

    for i, content_path in enumerate(input_files):
        content_file_name = os.path.basename(content_path)
        print(f"\n[{i + 1}/{len(input_files)}] {content_path}")

        abs_path = os.path.abspath(content_path)
        if abs_path in already_processed_paths:
            # In merge mode the file is already extracted; don't re-parse it. But still
            # honour --force-concept by overriding the stored category in place,
            # so re-running with --force-concept actually includes the file downstream.
            existing_key = path_to_key.get(abs_path)
            if args.force_concept and existing_key is not None:
                existing_cls = extracted_texts[existing_key].get("classification", {})
                if existing_cls.get("category") != "course_concept":
                    original = existing_cls.get("category")
                    print(f"--force-concept: overriding existing '{original}' -> 'course_concept'")
                    existing_cls.setdefault("original_category", original)
                    existing_cls["category"] = "course_concept"
                    extracted_texts[existing_key]["classification"] = existing_cls
                    persist_phase1()
            print(f"Already processed in this module, skipping extraction: {content_file_name}")
            continue

        # Extraction (paid/slow) + classification are wrapped so a failure on one file
        # neither crashes the run nor discards files already extracted in this run.
        try:
            text = extract_text_from_file(content_path, get_parser_service)
            if text is None:
                continue
            classification_json = classify_text(client, content_file_name, text, args.classification_model)
        except Exception as e:
            print(f"ERROR processing {content_file_name}: {e}. Skipping this file (already-saved files are preserved).")
            failed_files.append((content_file_name, str(e)))
            continue

        original_category = classification_json["category"]
        print(f"Classification: {original_category} ({classification_json.get('reason', '')})")

        if args.force_concept and original_category != "course_concept":
            print(f"--force-concept: overriding '{original_category}' -> 'course_concept'")
            classification_json["original_category"] = original_category
            classification_json["category"] = "course_concept"
        elif original_category != "course_concept":
            print("!" * 80)
            print(f"WARNING: '{content_file_name}' classified as '{original_category}', NOT 'course_concept'.")
            print("         Downstream generation will SKIP this file. Re-run with --force-concept to include it.")
            print("!" * 80)
            non_concept_files.append((content_file_name, original_category))

        key = unique_key(content_file_name, extracted_texts, content_path)
        extracted_texts[key] = {
            "file_path": content_path,
            "week_dir": week_dir_for(content_path),
            "text": text,
            "classification": classification_json,
        }
        already_processed_paths.add(abs_path)
        path_to_key[abs_path] = key

        # Persist after each file so an error on a later file doesn't lose this one.
        persist_phase1()

    persist_phase1()
    print(f"\nPhase 1 complete. Saved {len(extracted_texts)} file(s) to {json_path}")

    # ---------------------------------------------------------------
    # Phase 2: compression
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 2: Compress")
    print("=" * 80)

    to_compress = [
        (k, v) for k, v in extracted_texts.items()
        if args.overwrite or "compressed_text" not in v
    ]
    print(f"{len(to_compress)} file(s) to compress.")

    for i, (file_name, content_dict) in enumerate(to_compress):
        print(f"\n[{i + 1}/{len(to_compress)}] Compressing {file_name}")
        extracted_text = content_dict["text"]

        # Per-file resilience: a failure (e.g. bad model slug) skips this file rather than
        # aborting the whole phase. Already-compressed files remain saved on disk.
        try:
            compressed = compress_text(client, file_name, extracted_text, args.compression_model)
        except Exception as e:
            print(f"ERROR compressing {file_name}: {e}. Skipping this file.")
            failed_files.append((file_name, str(e)))
            continue

        # Guard: compression must never inflate. If the model's output is longer than the
        # original (e.g. already-dense text that only gains chunk labels / LaTeX formatting),
        # fall back to the raw extracted text so compressed_text is never larger than raw.
        if len(compressed) > len(extracted_text):
            print(f"Compressed ({len(compressed)}) > raw ({len(extracted_text)}); keeping raw extracted text instead.")
            compressed = extracted_text

        content_dict["compressed_text"] = compressed

        raw_len = len(extracted_text)
        compressed_len = len(compressed)
        pct = (float(raw_len - compressed_len) / raw_len * 100.0) if raw_len > 0 else 0.0
        print(f"Lengths | Raw: {raw_len} | Compressed: {compressed_len} | Decrease: {pct:.2f}%")

        # Persist incrementally so a crash mid-compression doesn't lose progress.
        save_module(json_path, module_data)

    write_module_metadata(json_path, extracted_texts)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Module: {module_name}")
    print(f"Total files in module: {len(extracted_texts)}")
    print(f"Output: {json_path}")

    if non_concept_files:
        print("\nThe following file(s) were NOT classified as 'course_concept' and will be")
        print("SKIPPED by the generation framework. Re-run with --force-concept to include them:")
        for fname, category in non_concept_files:
            print(f"  - {fname}  (classified: {category})")

    if failed_files:
        print("\nThe following file(s) FAILED during extraction/classification and were skipped:")
        for fname, err in failed_files:
            print(f"  - {fname}  ({err})")
