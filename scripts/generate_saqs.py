"""
This script generates synthetic short-answer questions (SAQs) based on extracted course content,
using Bloom's taxonomy levels to guide question complexity. It employs a two-stage process involving
question generation and verification to ensure quality and relevance. The generated questions,
along with their mark schemes and expected answers, are saved in a structured format for further use.
"""
import set_path
import argparse
import json
import os

from typing import List, Dict, Tuple, Set, Any
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict

from src.globals import (BASE_EXTRACTED_QUESTIONS_DATA_DIR, SYNTHETIC_ASSESSMENT_DATA_QUESTIONS_DIR)
from src.data_processing.synthetic.bloom_system_instructions import *
from src.data_processing.synthetic.generation_and_verification.saqs.generator import GeneratorSystem
from src.data_processing.synthetic.generation_and_verification.saqs.verifier import VerifierSystem
from src.data_processing.synthetic.pipelines.saqs import ZeroShotSAQGenerator, MultiStageZeroShotSAQGenerator, BloomSAQGenerator

# Models to generate (and benchmark) questions across. These slugs are sent to the
# OpenAI-compatible endpoint set by OPENAI_BASE_URL, so for OpenRouter they must be
# provider-namespaced. Override per run with --models.
DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4.1",
    "google/gemini-2.5-flash",
]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate synthetic SAQs across models and generation pipelines.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Model slugs to benchmark across (default: {DEFAULT_MODELS}). Use provider-namespaced slugs for OpenRouter, e.g. 'openai/gpt-4o-mini', 'google/gemini-2.5-flash'.")
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
        base_url=OPENAI_BASE_URL
    )

    # Models to benchmark across (override with --models)
    model_names = args.models
    print(f"Models: {model_names}")

    # Accumulates generated questions, keyed as: generator_name -> model_name -> file_name -> [questions]
    # (re-initialised per course in the loop below)
    results_per_generator = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Each JSON file under contents/ holds one course/module's extracted content
    module_json_files:List[str] = os.listdir(f"{BASE_EXTRACTED_QUESTIONS_DATA_DIR}/contents")
    print(f"Module JSON files found: {module_json_files}")
    module_json_paths = [f"{BASE_EXTRACTED_QUESTIONS_DATA_DIR}/contents/{file}" for file in module_json_files if file.endswith(".json")]

    for json_path in module_json_paths: # Iterate over courses
        with open(json_path, "r", encoding="utf-8") as f_json:
            module_data = json.load(f_json)
        
        # extracted_texts maps each source file name -> its extracted content + classification metadata
        extracted_texts_dict:Dict[str, Dict[str, str]] = module_data.get("extracted_texts", {})
        assert extracted_texts_dict is not None and isinstance(extracted_texts_dict, dict), "'extracted_texts' must be a dict"
        
        # Get course name for file naming
        course_name = os.path.splitext(os.path.basename(json_path))[0]
        
        # Reset results for this course
        results_per_generator = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        num_files_processed_for_course:int = 0

        # Skip files that are not classified as "course_concept"
        filtered_extracted_texts_dict = {file: content_dict for file, content_dict in extracted_texts_dict.items() if content_dict["classification"]["category"] == "course_concept"}
        print(f"Number of files: {len(filtered_extracted_texts_dict)}")

        # Generate questions for each file in the course
        for i, (file_name, content_dict) in enumerate(filtered_extracted_texts_dict.items()):
            # if num_files_processed_for_course == 3:
            #     print(f"Reached processing limit of 3 files for course {course_name}. Stopping further processing for this course.\n")
            #     break

            print(f"Processing file {i+1}/{len(filtered_extracted_texts_dict)}: {file_name}")
            compressed_text = content_dict.get("compressed_text") or content_dict.get("text")
            course_classification = content_dict.get("classification", {})

            # Skip files that are not classified as "course_concept"
            if course_classification.get("category", "") != "course_concept":
                print(f"Skipping file {file_name} as it is classified as '{course_classification.get('category', 'unknown')}'")
                continue

            for model_name in model_names:
                # Four baselines plus the proposed Bloom-aware pipeline, compared per file and model:
                # - zero_shot / zero_shot_bloom: single-call generation, without / with Bloom prompting
                # - multi_stage_zero_shot[_bloom]: staged generation, without / with Bloom prompting
                # - proposed_pipeline: Bloom-aware generation with per-stage verification
                question_generators = {
                    "zero_shot": ZeroShotSAQGenerator(client=client, model_name=model_name, use_bloom_prompting=False),
                    "zero_shot_bloom": ZeroShotSAQGenerator(client=client, model_name=model_name, use_bloom_prompting=True),
                    "multi_stage_zero_shot": MultiStageZeroShotSAQGenerator(client=client, model_name=model_name, use_bloom_prompting=False),
                    "multi_stage_zero_shot_bloom": MultiStageZeroShotSAQGenerator(client=client, model_name=model_name, use_bloom_prompting=True),
                    "proposed_pipeline": BloomSAQGenerator(client=client, model_name=model_name),
                }

                for generator_name, generator in question_generators.items():
                    print(f"Using question generator: {generator_name} (model={model_name}) for file: {file_name}")
                    # ensure generator gets the compressed text as input
                    content_for_generation = dict(content_dict)
                    content_for_generation["text"] = compressed_text

                    result:Dict[str, List[Dict[str, Any]]] = generator.generate_questions(
                        file_name=file_name,
                        content_dict=content_for_generation,
                    )

                    results_per_generator[generator_name][model_name][file_name].extend(result.get("valid_questions", []))

                    print(f"File: {file_name} | Model: {model_name} | Generator: {generator_name} | Num valid Q: {len(result.get('valid_questions', []))} | Num invalid Q: {len(result.get('failed_questions', []))}\n")

                    # Save immediately after each generator+model processes each file
                    # (flatten this generator's accumulated results into {model_name: [all questions across files]})
                    save_dict = {}
                    for m_name, file_dict in results_per_generator[generator_name].items():
                        all_questions = []
                        for fn, questions in file_dict.items():
                            all_questions.extend(questions)
                        save_dict[m_name] = all_questions

                    if save_dict:
                        save_path = f"{SYNTHETIC_ASSESSMENT_DATA_QUESTIONS_DIR}/saqs/{generator_name}___{course_name}.json"
                        os.makedirs(f"{SYNTHETIC_ASSESSMENT_DATA_QUESTIONS_DIR}/saqs", exist_ok=True)

                        with open(save_path, "w", encoding="utf-8") as f_out:
                            json.dump(save_dict, f_out, indent=4)

                        total_questions = sum(len(qs) for qs in save_dict.values())
                        print(f"Saved {total_questions} total questions from '{generator_name}' to: {save_path}")

            # Note: For only processing a few files per course.
            num_files_processed_for_course += 1

        # Final summary after processing all files in the course
        print(f"\n{'='*80}")
        print(f"Completed processing course: {course_name}")
        print(f"{'='*80}\n")
        
        for generator_name in results_per_generator.keys():
            for model_name, file_dict in results_per_generator[generator_name].items():
                total = sum(len(qs) for qs in file_dict.values())
                print(f"Generator name: {generator_name} | Model name: {model_name} | Total questions: {total}")