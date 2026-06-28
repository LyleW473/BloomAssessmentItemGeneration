import set_path
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict

from src.globals import (BASE_EXTRACTED_QUESTIONS_DATA_DIR)
from src.llm_response_generation.functions import (generate_llm_response, extract_json_from_text)

COMPRESSION_INSTRUCTION = """
You are an expert educational content preprocessing assistant.

This is NOT a summarisation task. You are performing high-fidelity compression of educational content.

Your task is to reduce the length of extracted course text while preserving the exact educational content needed for high-quality exam question generation.

The goal is NOT to summarise aggressively, simplify concepts, paraphrase heavily, or rewrite the material in a new style.
The goal is to produce a cleaned, compressed version of the input text that:
- removes irrelevant or low-value content,
- chunks the material into coherent exam-relevant sections,
- removes repeated information,
- preserves definitions, distinctions, examples, and examinable details,
- preserves mathematical notation and formulas clearly,
- applies lightweight LaTeX-style standardisation for consistency,
- and keeps wording as close as possible to the original source.

You must behave as a high-fidelity compression system, not as a generic summariser.

--------------------------------
INPUT FORMAT
--------------------------------
You will receive exactly one JSON object with this structure:

{
  "file_name": "<string>",
  "file_text": "<string>"
}

Field descriptions:
- "file_name": the name of the source file.
- "file_text": the extracted educational text to clean and compress.

Use "file_name" only as light contextual metadata if useful.
Your main task is to process "file_text".

--------------------------------
PRIMARY OBJECTIVE
--------------------------------
Given extracted educational text, produce a cleaned and compressed version that can later be used for question generation with minimal quality loss.

The output must:
1. truncate irrelevant sections,
2. chunk content intelligently,
3. deduplicate repeated content,
4. preserve definitions and key conceptual relationships,
5. keep wording similar or identical whenever possible,
6. preserve mathematical clarity and notation,
7. apply consistent lightweight LaTeX-style formatting where appropriate,
8. retain at least minimal intuition where it supports understanding.

--------------------------------
CORE RULES
--------------------------------

1. PRESERVE EXAMINABLE CONTENT
Keep all content that could plausibly support exam-style question generation, including:
- definitions,
- conceptual explanations,
- distinctions between related terms,
- methodological steps,
- comparisons,
- examples that clarify a concept,
- assumptions, limitations, advantages, disadvantages,
- cause-effect relationships,
- formulas or symbolic expressions,
- technical terminology,
- edge cases, caveats, and exceptions,
- short enumerations that express meaningful academic content.

Do not remove content merely because it seems detailed.
Detail should be preserved when it contributes to examinability.

2. REMOVE IRRELEVANT OR LOW-VALUE CONTENT
Remove or truncate content that is unlikely to help generate exam questions, such as:
- repeated introductory remarks,
- greetings, housekeeping text, timetable notes, admin reminders,
- repeated lecture signposting,
- filler transitions,
- generic motivational language,
- duplicate explanations of the same point,
- repeated examples that add no new information,
- long narrative padding around a concept,
- irrelevant metadata,
- references to slide order, file structure, or navigation,
- "today we will cover...", "as mentioned earlier...", "in the next slide...",
- purely conversational lecturer remarks that add no academic substance.

If a section contains both useful and irrelevant content, keep the useful part and remove only the irrelevant part.

3. DO NOT HEAVILY PARAPHRASE
Preserve original wording wherever possible.
Use exact phrases from the source whenever they are clear.
Only rewrite when necessary to:
- remove noise,
- join fragmented OCR/extracted text,
- fix obvious extraction artifacts,
- or improve local coherence after truncation.

Do not simplify terminology.
Do not replace precise academic wording with broader or vaguer wording.
Do not generalise specific claims.

4. PRESERVE DEFINITIONS EXACTLY OR NEAR-EXACTLY
Definitions are high priority.
When a definition appears, retain it with wording identical or very close to the source.
Do not compress definitions into vague shorthand.
If multiple related definitions appear, preserve the distinctions between them.

5. DEDUPLICATE CONSERVATIVELY
Remove repeated content only when it is genuinely redundant.
If two passages look similar but one includes additional nuance, detail, or a different example, preserve the richer version or merge them carefully.
Never remove content that changes the scope, specificity, or meaning of the material.

6. CHUNK INTELLIGENTLY
Organise the cleaned output into coherent chunks based on topic boundaries.

Each chunk should focus on one concept, method, relationship, comparison, or tightly related group of ideas.

Good chunking principles:
- one main topic per chunk,
- keep directly related definitions and explanations together,
- keep examples with the concept they illustrate,
- keep comparisons together,
- separate distinct topics clearly,
- avoid splitting a concept across chunks unless absolutely necessary.

7. MAINTAIN CONCEPTUAL RELATIONSHIPS
Preserve links such as:
- definition → explanation,
- concept → example,
- method → steps,
- model → strengths/weaknesses,
- technique → assumptions,
- problem → solution,
- comparison → differences,
- cause → effect.

Do not isolate concepts so aggressively that later question generation loses the surrounding academic context.

8. RETAIN ACADEMIC SPECIFICITY
Keep named methods, terms, models, algorithms, theories, datasets, frameworks, and technical vocabulary.
Preserve precise distinctions such as:
- supervised vs unsupervised,
- precision vs recall,
- correlation vs causation,
- training vs inference,
- bias vs variance.

9. HANDLE EXAMPLES CAREFULLY
Keep examples when they:
- clarify a definition,
- demonstrate application,
- illustrate a comparison,
- show limitations or edge cases,
- or are likely examinable.

Remove examples only if they are repetitive and clearly add no new value.

10. PRESERVE MATHEMATICAL AND SYMBOLIC CLARITY (WITH LIGHTWEIGHT LaTeX STANDARDISATION)
- Ensure formulas and notation remain readable and correct.
- Apply lightweight LaTeX-style formatting for consistency when appropriate.

Guidelines for standardisation:
- Use subscripts with underscores (e.g., C_k, x_i, y_i).
- Use \\hat{} for predictions where present (e.g., \\hat{y}_i).
- Use \\sum for summations and preserve index ranges clearly.
- Use ^ for exponents (e.g., x^2).
- Use \\| \\| for norms when applicable.
- Wrap expressions in inline LaTeX when helpful for clarity (e.g., $...$), but do not overuse.

Constraints:
- Do NOT introduce new symbols or notation not present in the source.
- Do NOT change the mathematical meaning.
- Do NOT over-complicate or fully rewrite equations.
- Only standardise existing expressions for clarity and consistency.

11. RETAIN MINIMAL INTUITION
Where available, preserve short intuitive explanations that:
- clarify a definition,
- explain why something works,
- or support interpretation.

Do not expand intuition, only retain it when present and useful.

12. DO NOT INVENT OR ADD CONTENT
- Use only information present in the source text.
- Do not infer missing claims.
- Do not add textbook knowledge.
- Do not introduce new examples or terminology.

--------------------------------
OUTPUT FORMAT
--------------------------------
Return exactly one JSON object with this structure:

{
  "compressed_text": "<string>"
}

The value of "compressed_text" must contain the cleaned and compressed content as a sequence of clearly labelled topical chunks inside a single string.

Use this internal chunk format inside the string:

[Chunk 1: <short topic label>]
<cleaned compressed content>

[Chunk 2: <short topic label>]
<cleaned compressed content>

...

Requirements:
- Each chunk label should be concise and topic-based.
- The chunk content should be clean, coherent, and compact.
- Preserve original wording as much as possible.
- Do not add commentary, explanations, or notes.
- Do not output justifications.
- Do not include any keys other than "compressed_text".
- Do not wrap the JSON in markdown fences.
- Output valid JSON only.

--------------------------------
COMPRESSION PRIORITY ORDER
--------------------------------
When deciding what to keep, prioritise in this order:

Highest priority:
1. definitions
2. distinctions between concepts
3. core explanations
4. key relationships
5. methodological steps
6. limitations / assumptions / caveats
7. examples that materially aid understanding

Lower priority:
8. repeated framing
9. repeated examples with no added value
10. administrative or conversational filler

--------------------------------
SPECIAL HANDLING RULES
--------------------------------

- If the input contains OCR noise or extraction artifacts, repair them minimally while preserving meaning.
- If a section is partially useful, keep the useful academic content and remove the surrounding noise.
- If a repeated concept appears across multiple areas, keep the clearest and most complete version.
- If exact wording contains examinable phrasing, preserve it.
- If a definition is short and precise, keep it verbatim if possible.
- If a long paragraph contains one important sentence and several filler sentences, keep the important sentence(s) and remove the filler.

--------------------------------
WHAT NOT TO DO
--------------------------------
Do NOT:
- produce a high-level summary,
- over-compress nuanced explanations,
- remove all examples,
- paraphrase definitions into generic language,
- collapse distinct concepts into one,
- rewrite the text into your own teaching style,
- change the meaning or specificity of the material.

--------------------------------
SUCCESS CRITERION
--------------------------------
The output should be shorter than the original, but still rich enough that a downstream model could generate precise, varied, and pedagogically valid exam questions without needing to consult the raw source text.

Aim to reduce the text significantly while preserving fidelity. Do not sacrifice important examinable content for additional compression.
"""

if __name__ == "__main__":
    # Load keys
    load_dotenv()
    OPENAI_LLM_API_KEY = os.getenv("OPENAI_LLM_API_KEY", None)
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)

    for key in [OPENAI_LLM_API_KEY, OPENAI_BASE_URL]:
        if key is None:
            raise ValueError(f"{key} not found in environment variables")
    

    client = OpenAI(
        api_key=OPENAI_LLM_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=60
    )
    
    model_name = "gemini-3-pro"

    # Load module data
    module_json_files:List[str] = os.listdir(f"{BASE_EXTRACTED_QUESTIONS_DATA_DIR}/contents")
    print(f"Module JSON files found: {module_json_files}")
    module_json_paths = [f"{BASE_EXTRACTED_QUESTIONS_DATA_DIR}/contents/{file}" for file in module_json_files if file.endswith(".json")]
    print(f"Module JSON paths: {module_json_paths}")

    for json_path in module_json_paths: # Iterate over courses
        with open(json_path, "r", encoding="utf-8") as f_json:
            module_data = json.load(f_json)
        
        print(f"{len(module_data['sorted_weeks_contents'])}")
        print(module_data['sorted_weeks_contents'])
        print(f"{len(module_data['extracted_texts'])}")
        extracted_texts_dict:Dict[str, Dict[str, str]] = module_data.get("extracted_texts", {})
        assert extracted_texts_dict is not None and isinstance(extracted_texts_dict, dict), "'extracted_texts' must be a dict"

        # # Skip files that are not classified as "course_concept" (Can change later if needed)
        # filtered_extracted_texts_dict = {file: content_dict for file, content_dict in extracted_texts_dict.items() if content_dict["classification"]["category"] == "course_concept"}
        # print(f"Number of files: {len(filtered_extracted_texts_dict)}")

        per_file_metadata = {}
        course_concept_stats = {"count": 0, "raw_total": 0, "compressed_total": 0}
        non_course_concept_stats = {"count": 0, "raw_total": 0, "compressed_total": 0}

        for i, (file_name, content_dict) in enumerate(extracted_texts_dict.items()):
            print(f"File: {i+1}/{len(extracted_texts_dict)}")

            extracted_text = content_dict["text"]

            # Classify the course content into "course_concept", "course_adjacent" or "administrative"
            compressed_text_input_json = {
                "file_name": file_name,
                "file_text": extracted_text
            }

            # Exponential retry with backoff in case the LLM response is None or invalid
            max_retries = 10
            base_delay = 1.0
            compressed_text_json = None

            for attempt in range(1, max_retries + 1):
                try:
                    compressed_text_response_text:str = generate_llm_response(
                        client=client,
                        model_name=model_name,
                        messages=[
                            {"role": "system", "content": COMPRESSION_INSTRUCTION},
                            {"role": "user", "content": json.dumps(compressed_text_input_json)}
                        ]
                    )

                    compressed_text_json = extract_json_from_text(compressed_text_response_text)

                    if compressed_text_json and isinstance(compressed_text_json, dict) and "compressed_text" in compressed_text_json:
                        break

                    raise ValueError("Invalid JSON payload or missing compressed_text")

                except Exception as e:
                    wait = base_delay * (2 ** (attempt - 1))
                    wait = min(wait, 30)
                    print(f"Attempt {attempt}/{max_retries} failed: {e}. Retrying in {wait:.1f}s...")
                    import time
                    time.sleep(wait)

            if not compressed_text_json or "compressed_text" not in compressed_text_json:
                raise RuntimeError(f"Failed to get valid compressed_text after {max_retries} attempts for {file_name}")

            compressed_text_as_str = compressed_text_json["compressed_text"]
            extracted_texts_dict[file_name]["compressed_text"] = compressed_text_as_str

            raw_len = len(extracted_text)
            compressed_len = len(compressed_text_as_str)
            pct_decrease = 0.0
            if raw_len > 0:
                pct_decrease = float(raw_len - compressed_len) / raw_len * 100.0

            # Track per-file metrics and category-specific aggregated stats
            category = content_dict.get("classification", {}).get("category", "unknown")
            per_file_metadata[file_name] = {
                "raw_length": raw_len,
                "compressed_length": compressed_len,
                "percentage_decrease": round(pct_decrease, 4),
                "classification_category": category
            }

            if category == "course_concept":
                course_concept_stats["raw_total"] += raw_len
                course_concept_stats["compressed_total"] += compressed_len
                course_concept_stats["count"] += 1
            else:
                non_course_concept_stats["raw_total"] += raw_len
                non_course_concept_stats["compressed_total"] += compressed_len
                non_course_concept_stats["count"] += 1

            print(f"Compressed text: {compressed_text_as_str}")
            print(f"Lengths | Raw: {raw_len} | Compressed text: {compressed_len} | Decrease: {pct_decrease:.2f}%")
            print()

        # Save updated module data back to the same source JSON file
        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump(module_data, f_json, indent=4, ensure_ascii=False)
        print(f"Saved compressed texts back to {json_path}")

        # Calculate overall metadata statistics
        overall_raw_total = sum(item["raw_length"] for item in per_file_metadata.values())
        overall_compressed_total = sum(item["compressed_length"] for item in per_file_metadata.values())
        overall_percentage_decrease = 0.0
        if overall_raw_total > 0:
            overall_percentage_decrease = float(overall_raw_total - overall_compressed_total) / overall_raw_total * 100.0

        # Compute per-category percentage decreases
        course_concept_percentage_decrease = 0.0
        if course_concept_stats["raw_total"] > 0:
            course_concept_percentage_decrease = float(course_concept_stats["raw_total"] - course_concept_stats["compressed_total"]) / course_concept_stats["raw_total"] * 100.0

        non_course_concept_percentage_decrease = 0.0
        if non_course_concept_stats["raw_total"] > 0:
            non_course_concept_percentage_decrease = float(non_course_concept_stats["raw_total"] - non_course_concept_stats["compressed_total"]) / non_course_concept_stats["raw_total"] * 100.0

        # Save per-course metadata JSON
        metadata_dir = os.path.join(BASE_EXTRACTED_QUESTIONS_DATA_DIR, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)

        module_metadata_path = os.path.join(metadata_dir, os.path.basename(json_path).replace('.json', '_metadata.json'))
        with open(module_metadata_path, 'w', encoding='utf-8') as f_meta:
            json.dump({
                "course_file": os.path.basename(json_path),
                "items": per_file_metadata,
                "overall_raw_total": overall_raw_total,
                "overall_compressed_total": overall_compressed_total,
                "overall_percentage_decrease": round(overall_percentage_decrease, 4),
                "course_concept_stats": {
                    "count": course_concept_stats["count"],
                    "raw_total": course_concept_stats["raw_total"],
                    "compressed_total": course_concept_stats["compressed_total"],
                    "percentage_decrease": round(course_concept_percentage_decrease, 4)
                },
                "non_course_concept_stats": {
                    "count": non_course_concept_stats["count"],
                    "raw_total": non_course_concept_stats["raw_total"],
                    "compressed_total": non_course_concept_stats["compressed_total"],
                    "percentage_decrease": round(non_course_concept_percentage_decrease, 4)
                }
            }, f_meta, indent=4, ensure_ascii=False)
        print(f"Saved metadata to {module_metadata_path}")