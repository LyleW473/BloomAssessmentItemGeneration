"""
This script generates synthetic scenario-based questions (SBQs) from extracted course content,
using Bloom's taxonomy levels to guide question complexity. For each course file it runs several
generators (zero-shot, multi-stage, and the proposed Bloom-aware pipeline) across multiple models,
verifying each generation stage. The generated questions, along with their scenarios, mark schemes,
and expected answers, are saved per generator in a structured JSON format for further use.
"""
import set_path
import argparse
import json
import os

import time
from typing import List, Dict, Tuple, Set, Any
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict
from src.globals import (BASE_EXTRACTED_QUESTIONS_DATA_DIR, SYNTHETIC_ASSESSMENT_DATA_QUESTIONS_DIR)
from src.data_processing.synthetic.bloom_system_instructions import *
from src.data_processing.synthetic.generation_and_verification.sbqs.generator import GeneratorSystem
from src.data_processing.synthetic.generation_and_verification.sbqs.verifier import VerifierSystem
from src.data_processing.synthetic.bloom_system_instructions.sbqs.generation import QUESTION_GEN_MAPPINGS
from src.data_processing.synthetic.pipelines.sbqs import BloomSBQGenerator, ZeroShotSBQGenerator, MultiStageZeroShotSBQGenerator

# Models to generate (and benchmark) questions across. These slugs are sent to the
# OpenAI-compatible endpoint set by OPENAI_BASE_URL, so for OpenRouter they must be
# provider-namespaced. Override per run with --models.
DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4.1",
    "google/gemini-2.5-flash",
]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate synthetic SBQs across models and generation pipelines.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Model slugs to benchmark across (default: {DEFAULT_MODELS}). Use provider-namespaced slugs for OpenRouter, e.g. 'openai/gpt-4o-mini', 'google/gemini-2.5-flash'.")
    parser.add_argument("--sbq-input", choices=["compressed", "raw"], default="compressed",
                        help="Text the SBQ generators are grounded in. 'compressed' (default) uses preprocessing's compressed_text; 'raw' uses the uncompressed extracted text to reproduce the published paper (which, due to a bug, fed raw text to every SBQ generator).")
    args = parser.parse_args()

    # Load keys
    load_dotenv()
    OPENAI_LLM_API_KEY = os.getenv("OPENAI_LLM_API_KEY", None)
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)

    for key in [OPENAI_LLM_API_KEY, OPENAI_BASE_URL]:
        if key is None:
            raise ValueError(f"{key} not found in environment variables")

    # Initialise OpenAI client
    client = OpenAI(
        api_key=OPENAI_LLM_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=60
    )

    # Models to benchmark across (override with --models)
    model_names = args.models
    print(f"Models: {model_names}")
    print(f"SBQ input text: {args.sbq_input}")

    # Load module data (each JSON file under contents/ holds one course/module's extracted content)
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
        # extracted_texts maps each source file name -> its extracted content + classification metadata
        extracted_texts_dict:Dict[str, Dict[str, str]] = module_data.get("extracted_texts", {})
        assert extracted_texts_dict is not None and isinstance(extracted_texts_dict, dict), "'extracted_texts' must be a dict"

        print(extracted_texts_dict.keys())

        # Get course name for file naming
        course_name = os.path.splitext(os.path.basename(json_path))[0]
        
        # Reset results for this course
        # Structure: generator_name -> model_name -> file_name -> [questions]
        results_per_generator = defaultdict(lambda: defaultdict(dict))

        # Start generating questions for each file
        course_questions:Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        num_files_processed_for_course:int = 0

        # Skip files that are not classified as "course_concept"
        filtered_extracted_texts_dict = {file: content_dict for file, content_dict in extracted_texts_dict.items() if content_dict["classification"]["category"] == "course_concept"}
        print(f"Number of files: {len(filtered_extracted_texts_dict)}")

        # Iterate over extracted texts in the module
        for i, (file_name, content_dict) in enumerate(filtered_extracted_texts_dict.items()):
            # if num_files_processed_for_course == 3:
            #     print(f"Reached processing limit of 3 files for course {course_name}. Stopping further processing for this course.\n")
            #     break

            print(f"Processing file {i+1}/{len(filtered_extracted_texts_dict)}: {file_name}")

            # Select the text the SBQ generators are grounded in.
            #   "compressed" (default): preprocessing's compressed_text (fallback to raw if absent) — corrected behavior.
            #   "raw": the uncompressed extracted text — reproduces the published paper.
            #
            # KNOWN BUG / REPRODUCIBILITY: the published SBQ results were generated on RAW text. The driver
            # passed the unmodified content_dict to every generator, and the SBQ pipelines read
            # content_dict["text"] (raw), so compressed_text was silently ignored for ALL SBQ generators.
            # The bug was uniform across pipelines, so the paper's pipeline-vs-pipeline SBQ comparison stays
            # valid; only the input text differed from intent. Use --sbq-input raw to reproduce the paper.
            if args.sbq_input == "compressed":
                input_text = content_dict.get("compressed_text") or content_dict.get("text")
            else:  # "raw"
                input_text = content_dict.get("text")

            # The SBQ pipelines read content_dict["text"], so inject the chosen text under that key
            # (mirrors generate_saqs.py).
            content_for_generation = dict(content_dict)
            content_for_generation["text"] = input_text

            for model_name in model_names:
                # Four baselines plus the proposed Bloom-aware pipeline, compared per file and model:
                # - zero_shot / zero_shot_bloom: single-call generation, without / with Bloom prompting
                # - multi_stage_zero_shot[_bloom]: staged generation, without / with Bloom prompting
                # - proposed_pipeline: Bloom-aware generation with per-stage verification
                question_generators = {
                    "zero_shot": ZeroShotSBQGenerator(client=client, model_name=model_name, use_bloom_prompting=False),
                    "zero_shot_bloom": ZeroShotSBQGenerator(client=client, model_name=model_name, use_bloom_prompting=True),
                    "multi_stage_zero_shot": MultiStageZeroShotSBQGenerator(client=client, model_name=model_name, use_bloom_prompting=False),
                    "multi_stage_zero_shot_bloom": MultiStageZeroShotSBQGenerator(client=client, model_name=model_name, use_bloom_prompting=True),
                    "proposed_pipeline": BloomSBQGenerator(client=client, model_name=model_name),
                }

                for generator_name, generator in question_generators.items():
                    print(f"Using question generator: {generator_name} for file: {file_name}")
                    result:Dict[str, List[Dict[str, Any]]] = question_generators[generator_name].generate_questions(
                        file_name=file_name,
                        content_dict=content_for_generation,
                    )
                    if file_name not in results_per_generator[generator_name][model_name]:
                        results_per_generator[generator_name][model_name][file_name] = []
                    results_per_generator[generator_name][model_name][file_name].extend(result["valid_questions"])

                    print(f"File: {file_name} | Num valid Q: {len(result['valid_questions'])} | Num invalid Q: {len(result['failed_questions'])}\n")

                    # Save immediately after each generator processes each file
                    # Flatten all accumulated results for this generator so far
                    save_dict = {}
                    for m, file_dict in results_per_generator[generator_name].items():
                        all_questions = []  
                        for fn, questions in file_dict.items():
                            all_questions.extend(questions)
                        save_dict[m] = all_questions

                    if save_dict: # Only save if there are questions
                        save_path = f"{SYNTHETIC_ASSESSMENT_DATA_QUESTIONS_DIR}/sbqs/{generator_name}___{course_name}.json"
                        os.makedirs(f"{SYNTHETIC_ASSESSMENT_DATA_QUESTIONS_DIR}/sbqs", exist_ok=True)
                        
                        with open(save_path, "w", encoding="utf-8") as f_out:
                            json.dump(save_dict, f_out, indent=4)
                        
                        total_saved_questions = sum(len(qs) for qs in save_dict.values())
                        print(f"Saved {total_saved_questions} total questions from '{generator_name}' to: {save_path}")


            # Note: For only processing a few files per course.
            num_files_processed_for_course += 1

        # Final summary after processing all files in the course
        print(f"\n{'='*80}")
        print(f"Completed processing course: {course_name}")
        print(f"{'='*80}\n")
        
        for generator_name in question_generators.keys():
            for model_name in results_per_generator[generator_name].keys():
                total = sum(
                    len(qs)
                    for qs in results_per_generator[generator_name][model_name].values()
                )
                print(f"Generator name: {generator_name} | Model name: {model_name} | Total questions: {total}")