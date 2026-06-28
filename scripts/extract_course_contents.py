import set_path
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any

from src.globals import (BASE_ASSESSMENT_QUESTIONS_DATA_DIR, BASE_EXTRACTED_QUESTIONS_DATA_DIR)
from src.data_processing.parser.service import ParserService
from src.data_processing.synthetic.utils import get_all_paths_in_dir, sort_contents_by_week_number
from src.llm_response_generation.functions import (generate_llm_response, extract_json_from_text)
from src.data_processing.synthetic.bloom_system_instructions import (
    COURSE_CONTENT_CLASSIFICATION_SYSTEM_INSTRUCTION
)

if __name__ == "__main__":
    # Load keys
    load_dotenv()
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", None)
    OPENAI_LLM_API_KEY = os.getenv("OPENAI_LLM_API_KEY", None)
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)

    model_name = "gpt-4o"
    for key in [OPENAI_LLM_API_KEY, OPENAI_BASE_URL, LLAMA_CLOUD_API_KEY]:
        if key is None:
            raise ValueError(f"{key} not found in environment variables")
    
    parser_service = ParserService(
        llama_cloud_api_key=LLAMA_CLOUD_API_KEY
    )
    client = OpenAI(
        api_key=OPENAI_LLM_API_KEY,
        base_url=OPENAI_BASE_URL
    )

    for course_dir in os.listdir(BASE_ASSESSMENT_QUESTIONS_DATA_DIR):
        print(f"Processing course: {course_dir}")
        if os.path.exists(f"{BASE_EXTRACTED_QUESTIONS_DATA_DIR}/contents/{course_dir}.json"):
            print(f"Extracted data for course {course_dir} already exists, skipping...")
            continue
        
        week_contents = os.listdir(f"{BASE_ASSESSMENT_QUESTIONS_DATA_DIR}/{course_dir}")
        sorted_week_contents = sort_contents_by_week_number(week_contents)

        extracted_texts = {}
        for week_dir in sorted_week_contents:
            all_content_paths:List[str] = get_all_paths_in_dir(f"{BASE_ASSESSMENT_QUESTIONS_DATA_DIR}/{course_dir}/{week_dir}")

            for content_path in all_content_paths:
                content_file_name = os.path.basename(content_path)
                print(f"Course: {course_dir} | Week: {week_dir} | File name: {content_file_name}")

                if content_file_name.endswith(".txt"):
                    # For text files, just read the content directly
                    with open(content_path, "r", encoding="utf-8") as f_txt:
                        result_txt = f_txt.read()
                    
                elif content_file_name.endswith(".mp4"): # TODO: Add video parsing for transcripts
                    continue
                
                else:
                    # Parse the file to extract text content (LlamaParse takes the real path directly)
                    result:Dict[str, Any] = parser_service.parse_material(
                        file_path=content_path,
                        subject=course_dir,
                    )

                    if result["status"] == "error":
                        print(f"Error parsing file {content_file_name}: {result['error']}")
                        continue

                    result_txt = result["texts"]

                    print(json.dumps(result, indent=4))
                    print()
                
                # Classify the course content into "course_concept", "course_adjacent" or "administrative"
                classification_input_json = {
                    "file_name": content_file_name,
                    "file_text": result_txt
                }
                classification_response_text:str = generate_llm_response(
                    client=client,
                    model_name=model_name,
                    messages=[
                        {"role": "system", "content": COURSE_CONTENT_CLASSIFICATION_SYSTEM_INSTRUCTION},
                        {"role": "user", "content": f"Classify the following course content into one of the specified categories.\n\n" + json.dumps(classification_input_json)}
                    ]
                )
                classification_json = extract_json_from_text(classification_response_text)
                print(f"Course content classification response JSON:\n {json.dumps(classification_json, indent=4)}")
                print()
                assert classification_json is not None and isinstance(classification_json, dict), "Course content classification JSON is not a dict"
                assert "category" in classification_json, "Course content classification JSON missing 'category' field"
                assert classification_json["category"] in ["course_concept", "course_adjacent", "administrative"], "Invalid classification value"
                
                # Add to the extracted texts
                extracted_texts[content_file_name] = {
                    "file_path": content_path,
                    "week_dir": week_dir,
                    "text": result_txt,
                    "classification": classification_json
                }

        data_to_save = {
            "sorted_weeks_contents": sorted_week_contents,
            "extracted_texts": extracted_texts
        }

        # Save the extracted texts to a JSON file for later use
        os.makedirs(f"{BASE_EXTRACTED_QUESTIONS_DATA_DIR}/contents", exist_ok=True)
        with open(f"{BASE_EXTRACTED_QUESTIONS_DATA_DIR}/contents/{course_dir}.json", "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4)