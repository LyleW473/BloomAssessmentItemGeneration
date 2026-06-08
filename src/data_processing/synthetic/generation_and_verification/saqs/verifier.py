"""
SAQ (short-answer question) verifier system.
- Defines `VerifierSystem`, which extends the base phase-one checks with SAQ-specific verification.
- Adds the mark-scheme coverage and expected-answer coverage checks (verification stages 3 and 4).
"""
import json
from openai import OpenAI
from typing import List, Dict, Any
from src.llm_response_generation.functions import (generate_llm_json_response)
from src.data_processing.synthetic.bloom_system_instructions import (
    MARK_SCHEME_COVERAGE_VERIFICATION_SYSTEM_INSTRUCTION,
    EXPECTED_ANSWER_COVERAGE_VERIFICATION_SYSTEM_INSTRUCTION
)
from src.data_processing.synthetic.generation_and_verification.base_verifier_system import BaseVerifierSystem

class VerifierSystem(BaseVerifierSystem):
    """
    A class to perform verification checks on generated synthetic assessment questions and related content using LLMs.

    This class encapsulates methods to verify various aspects of the generated content, including:
    - Bloom's taxonomy level alignment for questions.
    - Relevance of questions to the course content.

    + (unique for SAQs):
    - Coverage of mark schemes against generated questions.
    - Coverage of expected answers against generated mark schemes.
    """

    def __init__(self, client:OpenAI, model_name:str):
        """
        Initalises the VerifierSystem with the given OpenAI client and model name.

        Args:
            client (OpenAI): The OpenAI client instance for making API calls.
            model_name (str): The name of the model to use for verification tasks.
        """
        super().__init__(client, model_name)
        
        # Add on top of the base stages
        self.stages[3] = "mark_scheme_coverage_check (3)"
        self.stages[4] = "expected_answer_coverage_check (4)"

    def perform_mark_scheme_coverage_check(
            self,
            question_dict:Dict[str, str],
            mark_scheme_dict:Dict[str, Any],
            bloom_level:str,
            difficulty:str,
            extracted_text:str,
            file_name:str,
        ) -> List[Dict[str, str]]:
        """
        Performs the mark scheme coverage verification check.
        - Check: "Is the generated mark scheme factually accurate and relevant. Do the points mentioned in the mark scheme accurately reflect key concepts from the course content?"

        Args:
            question_dict (Dict[str, str]): A dictionary containing the generated question details.
            mark_scheme_dict (Dict[str, Any]): A dictionary containing the generated mark scheme details.
            bloom_level (str): The Bloom's taxonomy level associated with the question.
            difficulty (str): The difficulty level associated with the question.
            extracted_text (str): The extracted course content text used for verification.
            file_name (str): The name of the file from which the question and mark scheme were generated (for logging purposes).

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing details of questions that failed the mark scheme coverage verification check.
                                  (Should either be an empty list if the mark scheme passed the check, or a list with one dict if it failed.)
        """
            
        mark_scheme_coverage_check_input_json = {
            "question": question_dict["question"],
            "mark_scheme": mark_scheme_dict["mark_scheme"],
            "bloom_level": bloom_level,
            "difficulty": difficulty,
            "course_content": extracted_text
        }
        mark_scheme_coverage_prompt_text = f"Evaluate whether the following mark scheme accurately reflects key concepts from the course content provided earlier.\n\n" + json.dumps(mark_scheme_coverage_check_input_json)
        mark_scheme_coverage_json = generate_llm_json_response(
            client=self.client,
            model_name=self.model_name,
            system_instruction=MARK_SCHEME_COVERAGE_VERIFICATION_SYSTEM_INSTRUCTION,
            prompt_text=mark_scheme_coverage_prompt_text
        )
        # print(f"Mark scheme coverage verification response JSON:\n {json.dumps(mark_scheme_coverage_json, indent=4)}")
        # print(f"Original mark scheme JSON:\n {json.dumps(mark_scheme_json, indent=4)}")
        # print()
        try:
            assert mark_scheme_coverage_json is not None and isinstance(mark_scheme_coverage_json, dict), "Mark scheme coverage verification JSON is not a dict"
            assert "is_correct" in mark_scheme_coverage_json, "Mark scheme coverage verification JSON missing 'is_correct' field"
            assert "reason" in mark_scheme_coverage_json, "Mark scheme coverage verification JSON missing 'reason' field"
            assert isinstance(mark_scheme_coverage_json["is_correct"], bool), "'is_correct' field must be a boolean"
            assert isinstance(mark_scheme_coverage_json["reason"], str), "'reason' field must be a string"
        except Exception as e:
            # print(f"Error in mark scheme coverage verification JSON: {e}")
            return []

        # Check if the mark scheme failed the coverage verification
        if mark_scheme_coverage_json["is_correct"] is False:
            info_dict = self._generate_info_dict(
                file_name=file_name,
                question_dict=question_dict,
                mark_scheme_dict=mark_scheme_dict,
                answer_dict=None,
                verification_failure_stage=3,
                corresponding_verification_dict=mark_scheme_coverage_json
            )
            # print(f"Mark scheme failed coverage verification check. Skipping further processing for this question.\n")
            # print(json.dumps(info_dict, indent=4))
            return [info_dict]
        return []
    
    def perform_expected_answer_coverage_check(
            self,
            question_dict:Dict[str, str],
            mark_scheme_dict:Dict[str, Any],
            answer_dict:Dict[str, Any],
            file_name:str,
        ) -> List[Dict[str, str]]:
        """
        Performs the expected answer coverage verification check.
        - Check: "Does the expected answer comprehensively cover all points in the mark scheme? Is the answer well-written, clear, and academically appropriate?"

        Args:
            question_dict (Dict[str, str]): A dictionary containing the generated question details.
            mark_scheme_dict (Dict[str, Any]): A dictionary containing the generated mark scheme details.
            answer_dict (Dict[str, Any]): A dictionary containing the generated expected answer details.
            file_name (str): The name of the file from which the question, mark scheme, and expected answer were generated (for logging purposes).
        Returns:
            List[Dict[str, str]]: A list of dictionaries containing details of questions that failed the expected answer coverage verification check.
                                  (Should either be an empty list if the expected answer passed the check, or a list with one dict if it failed.)
        """

        expected_answer_coverage_input_json = {
            "question": question_dict["question"],
            "mark_scheme": mark_scheme_dict["mark_scheme"],
            "expected_answer": answer_dict["expected_answer"],
        }
        expected_answer_coverage_prompt_text = f"Evaluate whether the following expected answer comprehensively covers all points in the mark scheme provided earlier.\n\n" + json.dumps(expected_answer_coverage_input_json)
        expected_answer_coverage_json = generate_llm_json_response(
            client=self.client,
            model_name=self.model_name,
            system_instruction=EXPECTED_ANSWER_COVERAGE_VERIFICATION_SYSTEM_INSTRUCTION,
            prompt_text=expected_answer_coverage_prompt_text
        )
        # print(f"Expected answer coverage verification response JSON:\n {json.dumps(expected_answer_coverage_json, indent=4)}")
        # print(f"Original expected answer JSON:\n {json.dumps(answer_dict, indent=4)}")
        # print()
        try:
            assert expected_answer_coverage_json is not None and isinstance(expected_answer_coverage_json, dict), "Expected answer coverage verification JSON is not a dict"
            assert "is_covered" in expected_answer_coverage_json, "Expected answer coverage verification JSON missing 'is_covered' field"
            assert "reason" in expected_answer_coverage_json, "Expected answer coverage verification JSON missing 'reason' field"
            assert isinstance(expected_answer_coverage_json["is_covered"], bool), "'is_covered' field must be a boolean"
            assert isinstance(expected_answer_coverage_json["reason"], str), "'reason' field must be a string"
        except Exception as e:
            # print(f"Error in expected answer coverage verification JSON: {e}")
            return []

        # Check if the expected answer failed the coverage verification
        if expected_answer_coverage_json["is_covered"] is False:
            info_dict = self._generate_info_dict(
                file_name=file_name,
                question_dict=question_dict,
                mark_scheme_dict=mark_scheme_dict,
                answer_dict=answer_dict,
                verification_failure_stage=4,
                corresponding_verification_dict=expected_answer_coverage_json
            )
            # print(f"Expected answer failed coverage verification check. Skipping further processing for this question.\n")
            # print(json.dumps(info_dict, indent=4))
            return [info_dict]
        return []