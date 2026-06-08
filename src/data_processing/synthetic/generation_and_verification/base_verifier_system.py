"""
Base verifier system shared by the assessment question-generation pipelines.
- Defines `BaseVerifierSystem`, which runs the first-phase LLM verification checks on generated questions.
- Phase one covers Bloom's-taxonomy-level alignment and course-content relevance; subclasses add the later-stage coverage checks.
"""
import json
from openai import OpenAI
from typing import List, Dict, Any, Union, Tuple, Set
from src.llm_response_generation.functions import (generate_llm_json_response)
from src.data_processing.synthetic.bloom_system_instructions import (
    BLOOM_LEVEL_VERIFICATION_SYSTEM_INSTRUCTION,
    COURSE_RELEVANCE_VERIFICATION_SYSTEM_INSTRUCTION,
)

class BaseVerifierSystem:
    """
    A base class to perform verification checks on generated synthetic assessment questions and related content using LLMs.

    This class encapsulates methods to verify various aspects of the generated content, including:
    - Bloom's taxonomy level alignment for questions.
    - Relevance of questions to the course content.
    """

    def __init__(self, client:OpenAI, model_name:str):
        """
        Initalises the VerifierSystem with the given OpenAI client and model name.

        Args:
            client (OpenAI): The OpenAI client instance for making API calls.
            model_name (str): The name of the model to use for verification tasks.
        """
        self.client = client
        self.model_name = model_name

        self.stages = {
            1: "bloom_level_check (1)",
            2: "course_relevance_check (2)",
        }

    def _generate_info_dict(
            self,
            file_name:str,
            question_dict:Dict[str, str],
            mark_scheme_dict:Union[Dict[str, Any], None],
            answer_dict:Union[Dict[str, Any], None],
            verification_failure_stage:int,
            corresponding_verification_dict:Dict[str, Any]
        ):
        """
        Generates an information dictionary encapsulating details about failed verification checks.

        Args:
            file_name (str): The name of the file from which the question, mark scheme, and expected answer were generated.
            question_dict (Dict[str, str]): A dictionary containing the generated question details.
            mark_scheme_dict (Union[Dict[str, Any], None]): A dictionary containing the generated mark scheme details, or None if not applicable (e.g., not reached that stage).
            answer_dict (Union[Dict[str, Any], None]): A dictionary containing the generated expected answer details, or None if not applicable (e.g., not reached that stage).
            verification_failure_stage (int): The stage number at which the verification failed (1 to 4).
            corresponding_verification_dict (Dict[str, Any]): The verification result dictionary corresponding to the failed check.
        
        Returns:
            Dict[str, Any]: A dictionary containing all relevant information about the failed verification.
        """
        info_dict = {
                "file_name": file_name,
                "question_dict": question_dict,
                "mark_scheme_dict": mark_scheme_dict,
                "answer_dict": answer_dict,
                "verification_stage": self.stages[verification_failure_stage],
                "corresponding_verification_dict": corresponding_verification_dict
            }
        return info_dict


    def perform_phase_one_check(
            self,
            question_json:List[Dict[str, str]],
            extracted_text:str,
            file_name:str,
            skip_course_relevance_check:bool=False
        ) -> Tuple[List[Dict[str, str]], Set[int]]:
        """
        Performs the first phase of verification checks for the entire synthetic generation process.
        - This first phase focuses on verifying the generated question stems only.
        - Check 1: Verify if each generated question aligns with the specified Bloom's taxonomy level.
        - Check 2: Verify if each generated question is relevant to the course content.

        Args:
            question_json (List[Dict[str, str]]): A list of generated question dictionaries to verify. Each dictionary contains keys like "question", "bloom_level", etc.
            extracted_text (str): The extracted course content text used for relevance verification.
            file_name (str): The name of the file from which the questions were generated (for logging purposes).
        Returns:
            additional_failed_questions (List[Dict[str, str]]): A list of dictionaries containing details of questions that failed the verification checks.
            failed_indices (Set[int]): A set of indices of questions that failed any of the checks (so they can be filtered out later on).
        """

        additional_failed_questions:List[Dict[str, str]] = []
        failed_indices = set() # Indices of questions that failed any of the checks (so we can filter them out later on)

        for question_idx, (generated_question) in enumerate(question_json): # Iterate through each generated question
            
            # Check 1: "Does the generated question align with the specified Bloom's taxonomy level?"
            bloom_level_check_input_json = {
                "question": generated_question["question"],
                "bloom_level": generated_question["bloom_level"]
            }
            bloom_level_verification_prompt_text = f"Verify whether the following question matches the specified Bloom's taxonomy level.\n\n" + json.dumps(bloom_level_check_input_json)
            bloom_level_verification_json = generate_llm_json_response(
                client=self.client,
                model_name=self.model_name,
                system_instruction=BLOOM_LEVEL_VERIFICATION_SYSTEM_INSTRUCTION,
                prompt_text=bloom_level_verification_prompt_text
            )

            # print(f"Bloom level verification response JSON:\n {json.dumps(bloom_level_verification_json, indent=4)}")
            # print(f"Original question JSON:\n {json.dumps(generated_question, indent=4)}")
            try:
                assert bloom_level_verification_json is not None and isinstance(bloom_level_verification_json, dict), "Bloom level verification JSON is not a dict"
                assert "belongs_to_level" in bloom_level_verification_json, "Bloom level verification JSON missing 'belongs_to_level' field"
                assert "reason" in bloom_level_verification_json, "Bloom level verification JSON missing 'reason' field"
                assert isinstance(bloom_level_verification_json["belongs_to_level"], bool), "'belongs_to_level' field must be a boolean"
                assert isinstance(bloom_level_verification_json["reason"], str), "'reason' field must be a string"
            except Exception as e:
                # print(f"Error in Bloom level verification JSON: {e}")
                continue # If the verification response JSON is malformed, skip to the next question without marking this one as failed since we can't be sure if it truly failed or if there was just an issue with the verification response format.

            # Check if the question failed the Bloom level verification
            if bloom_level_verification_json["belongs_to_level"] is False:
                info_dict = self._generate_info_dict(
                    file_name=file_name,
                    question_dict=generated_question,
                    mark_scheme_dict=None,
                    answer_dict=None,
                    verification_failure_stage=1,
                    corresponding_verification_dict=bloom_level_verification_json
                )
                additional_failed_questions.append(info_dict)
                failed_indices.add(question_idx)
                # print(f"Question failed Bloom level verification check. Skipping further processing for this question.\n")
                # print(json.dumps(info_dict, indent=4))
            
            # Check 2: "Is the generated question relevant to the course in a meaningful way?"
            if skip_course_relevance_check:
                # print(f"Skipping course relevance check for question index {question_idx} as per flag.")
                continue
            
            # Note: This is because there are some cases where the uploaded materials is not relevant e.g., "Who was the module leader for this course?"
            course_relevance_input_json = {
                "question": generated_question["question"],
                "course_content": extracted_text
            }
            course_relevance_verification_prompt_text = f"Verify whether the following question is relevant to the course content provided earlier.\n\n" + json.dumps(course_relevance_input_json)
            course_relevance_verification_json = generate_llm_json_response(
                client=self.client,
                model_name=self.model_name,
                system_instruction=COURSE_RELEVANCE_VERIFICATION_SYSTEM_INSTRUCTION,
                prompt_text=course_relevance_verification_prompt_text
            )

            # print(f"Course relevance verification response JSON:\n {json.dumps(course_relevance_verification_json, indent=4)}")
            # print(f"Original question JSON:\n {json.dumps(generated_question, indent=4)}")
            # print()

            # TODO: Improve question generation prompt by explicitly stating to avoid questions that are not relevant to the course content.
            try:
                assert course_relevance_verification_json is not None and isinstance(course_relevance_verification_json, dict), "Course relevance verification JSON is not a dict"
                assert "is_relevant" in course_relevance_verification_json, "Course relevance verification JSON missing 'is_relevant' field"
                assert "reason" in course_relevance_verification_json, "Course relevance verification JSON missing 'reason' field"
                assert isinstance(course_relevance_verification_json["is_relevant"], bool), "'is_relevant' field must be a boolean"
                assert isinstance(course_relevance_verification_json["reason"], str), "'reason' field must be a string"
            except Exception as e:
                # print(f"Error in course relevance verification JSON: {e}")
                continue
                
            # Check if the question failed the course relevance verification
            if course_relevance_verification_json["is_relevant"] is False:
                info_dict = self._generate_info_dict(
                    file_name=file_name,
                    question_dict=generated_question,
                    mark_scheme_dict=None,
                    answer_dict=None,
                    verification_failure_stage=2,
                    corresponding_verification_dict=course_relevance_verification_json
                )
                additional_failed_questions.append(info_dict)
                failed_indices.add(question_idx)
                # print(f"Question failed course relevance verification check. Skipping further processing for this question.\n")
                # print(json.dumps(info_dict, indent=4))

        return additional_failed_questions, failed_indices