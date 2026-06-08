"""
SAQ (short-answer question) generator system.
- Defines `GeneratorSystem`, which generates mark schemes and expected answers for short-answer questions.
- Extends `BaseGeneratorSystem` (Bloom-aware question-stem generation) with the SAQ-specific generation stages.
"""
import json
from typing import List, Dict, Any
from openai import OpenAI
from src.llm_response_generation.functions import generate_llm_json_response
from src.data_processing.synthetic.bloom_system_instructions.shared import (
    MARK_SCHEME_GENERATION_SYSTEM_INSTRUCTION,
    ANSWER_GENERATION_SYSTEM_INSTRUCTION
)
from src.data_processing.synthetic.generation_and_verification.base_generator_system import BaseGeneratorSystem

class GeneratorSystem(BaseGeneratorSystem):
    """
    Generator system for short-answer questions (SAQs).
    - Inherits Bloom-aware question-stem generation from `BaseGeneratorSystem`.
    - Adds the SAQ-specific stages: mark-scheme generation and expected-answer generation.
    """

    def __init__(self, client:OpenAI, model_name:str):
        """
        Initialises the SAQ GeneratorSystem with the given OpenAI client and model name.

        Args:
            client (OpenAI): The OpenAI client instance that supports LLM responses for multiple models including Gemini.
            model_name (str): The name of the model to use for generation (must be supported by the endpoint).
        """
        super().__init__(client, model_name)
    
    def generate_mark_scheme(
            self,
            question:str,
            difficulty:str,
            bloom_level:str,
            extracted_text:str
        ) -> Dict[str, Any]:
        """
        Generates a mark scheme for a given question based on its difficulty, Bloom level, and extracteed course content.

        Args:
            question (str): The assessment question for which to generate the mark scheme.
            difficulty (str): The difficulty level of the question (e.g., "easy", "medium", "hard") [derived from Bloom level].
            bloom_level (str): The Bloom's taxonomy level of the question (e.g., "knowledge", "understanding", "application", etc.).
            extracted_text (str): The extracted course content text to base the mark scheme on.

        Returns:
            Dict[str, Any]: A dictionary containing the question used to generate the mark scheme and the generated mark scheme.
                            The mark scheme dict contains a list of marking points with their corresponding marks and the total marks.
                            e.g.,
                            {
                                "question": "Define what a training set is in supervised learning.",
                                "mark_scheme": [
                                    {
                                        "point": "A training set is a portion of the data used for learning.",
                                        "marks": 1
                                    },
                                    {
                                        "point": "In supervised learning, the training set includes data and labels.",
                                        "marks": 1
                                    },
                                    {
                                        "point": "The algorithm learns from the labeled portion of the training set.",
                                        "marks": 1
                                    }
                                ],
                                "total_marks": 3
                            }
        """
        mark_scheme_generation_input_json = {
            "question": question,
            "difficulty": difficulty,
            "bloom_level": bloom_level,
            "course_content": extracted_text
        }
        mark_scheme_generation_prompt_text = f"Generate a JSON object following the schema described in the system message. \n\n" + json.dumps(mark_scheme_generation_input_json)
        mark_scheme_json = generate_llm_json_response(
            client=self.client,
            model_name=self.model_name,
            system_instruction=MARK_SCHEME_GENERATION_SYSTEM_INSTRUCTION,
            prompt_text=mark_scheme_generation_prompt_text
        )
        if mark_scheme_json is None:
            return None
        mark_scheme_json["total_marks"] = sum(point["marks"] for point in mark_scheme_json.get("mark_scheme", [])) # Derive "total_marks" directly from the sum of marks in "mark_scheme"
        # print(f"Parsed mark scheme JSON:\n {json.dumps(mark_scheme_json, indent=4)}")
        
        try:
            assert mark_scheme_json is not None and isinstance(mark_scheme_json, dict), "Mark scheme JSON is not a dict"
            assert "mark_scheme" in mark_scheme_json and "total_marks" in mark_scheme_json, "Mark scheme JSON missing required fields"
            assert isinstance(mark_scheme_json["mark_scheme"], list) and all(isinstance(point, dict) and "point" in point and "marks" in point for point in mark_scheme_json["mark_scheme"]), "Mark scheme must be a list of dicts with 'point' and 'marks'"
            assert isinstance(mark_scheme_json["total_marks"], int) and mark_scheme_json["total_marks"] == sum(point["marks"] for point in mark_scheme_json["mark_scheme"]), "Total marks must equal sum of marks in mark scheme"
        except Exception as e:
            return None
        
        # print(f"Generated mark scheme: {json.dumps(mark_scheme_json, indent=4)}")
        return mark_scheme_json
    
    def generate_expected_answer(
        self,
        question:str,
        difficulty:str,
        mark_scheme_json:Dict[str, Any]
        ) -> Dict[str, Any]:
        """
        Generates an expected answer for a given question based on its difficulty and the provided mark scheme.

        Args:
            question (str): The assessment question for which to generate the expected answer.
            difficulty (str): The difficulty level of the question (e.g., "easy", "medium", "hard") [derived from Bloom level].
            mark_scheme_json (Dict[str, Any]): The mark scheme dictionary containing the the question and the generated mark scheme (see 'generate_mark_scheme' method for schema).
        Returns:
            Dict[str, Any]: A dictionary containing the expected answer for the question.
                            e.g.,
                            {
                                "question": "Define what a training set is in the context of supervised learning.",
                                "expected_answer": "In supervised learning, a training set is the portion of data utilized for learning by providing examples that the algorithm can 
                                                    use to infer patterns. It consists of data with corresponding labels, allowing the program to learn the relationship between input 
                                                    and output. This foundational dataset is essential for constructing models that can predict or classify new data accurately."
                            }

        """
        answer_generation_input_json = {
            "question": question,
            "difficulty": difficulty,
            "mark_scheme": mark_scheme_json["mark_scheme"]
        }
        answer_generation_prompt_text = f"Generate a JSON object following the schema described in the system message. \n\n" + json.dumps(answer_generation_input_json)
        answer_json = generate_llm_json_response(
            client=self.client,
            model_name=self.model_name,
            system_instruction=ANSWER_GENERATION_SYSTEM_INSTRUCTION,
            prompt_text=answer_generation_prompt_text
        )
        if answer_json is None:
            return None
        # print(f"Parsed answer JSON:\n {json.dumps(answer_json, indent=4)}")

        try:
            assert answer_json is not None and isinstance(answer_json, dict), "Answer JSON is not a dict"
            assert "expected_answer" in answer_json, "Answer JSON missing required fields"
            assert isinstance(answer_json["expected_answer"], str) and len(answer_json["expected_answer"].strip()) > 0, "Expected answer must be a non-empty string"
        except Exception as e:
            return None
        
        # print(f"Generated expected answer: {json.dumps(answer_json, indent=4)}")
        return answer_json