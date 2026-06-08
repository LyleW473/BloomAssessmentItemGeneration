"""
Base generator system shared by the assessment question-generation pipelines.
- Defines `BaseGeneratorSystem`, which produces Bloom-aware question stems from extracted course content.
- Subclassed by the SAQ generator system (and, through it, the SBQ generator system), which add format-specific generation steps (e.g. mark schemes, expected answers).
"""
import json
from typing import List, Dict
from openai import OpenAI
from src.llm_response_generation.functions import generate_llm_json_response
from src.data_processing.synthetic.bloom_system_instructions import (
    BLOOM_LEVEL_TO_DIFFICULTY_MAPPING
)
class BaseGeneratorSystem:
    """
    Base class for generating assessment question stems from course content using an LLM.
    - Holds the shared OpenAI client and model name used for every generation call.
    - Provides `generate_question_stems`, the shared question-stem generation step used by the SAQ/SBQ generator systems.
    """

    def __init__(self, client:OpenAI, model_name:str):
        """
        Initialises the BaseGeneratorSystem with the given OpenAI client and model name.

        Args:
            client (OpenAI): The OpenAI client instance that supports LLM responses for multiple models including Gemini.
            model_name (str): The name of the model to use for generating questions (must be supported by the endpoint).
        """
        self.client = client
        self.model_name = model_name

    # Shared question-stem generation step used by the SAQ/SBQ generator systems
    def generate_question_stems(
            self,
            extracted_text:str,
            bloom_level:str,
            question_gen_system_instruction:str,
        ) -> List[Dict[str, str]]:
        """
        Generates assessment question stems based on the provided course content.

        Args:
            extracted_text (str): The extracted course content text to base the questions on.
            bloom_level (str): The Bloom's taxonomy level to guide question generation.
            question_gen_system_instruction (str): The system instruction to guide the question generation process (this should be a Bloom-level aware specific instruction, but can be customized as needed).
        
        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing a dict (which is a generated question dict containing the question text and its assigned Bloom level).
                                  e.g., [
                                            {
                                                "question": "State what precision is in classification metrics.",
                                                "bloom_level": "knowledge"
                                            },
                                            {
                                                "question": "Define sensitivity in the context of classification problems.",
                                                "bloom_level": "knowledge"
                                            },
                                            ...
                                            {
                                                "question": "List the components used to calculate the F1 score.",
                                                "bloom_level": "knowledge"
                                            }
                                        ]
        """
        question_gen_prompt_text = json.dumps({
            "extracted_text": extracted_text,
            "bloom_level": bloom_level
        })
        question_json = generate_llm_json_response(
            client=self.client,
            model_name=self.model_name,
            system_instruction=question_gen_system_instruction,
            prompt_text=question_gen_prompt_text
        )
        try:
            assert question_json is not None and isinstance(question_json, list), "Generated JSON is not a list"
            assert all(isinstance(q, dict) and "question" in q and "bloom_level" in q for q in question_json), "Each question entry must be a dict with 'question' and 'bloom_level'"

            # Check that the "bloom_level" is one of the expected levels of Bloom's taxonomy
            for q in question_json:
                bloom_level = q["bloom_level"].strip().lower()
                difficulty = BLOOM_LEVEL_TO_DIFFICULTY_MAPPING.get(bloom_level, None)
                assert difficulty is not None, f"Unrecognized bloom level '{bloom_level}' in question dict: {q}"
        except:
            return None

        # print(json.dumps(question_json, indent=4))
        return question_json