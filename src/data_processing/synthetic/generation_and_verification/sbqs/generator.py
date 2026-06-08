"""
SBQ (scenario-based question) generator system.
- Defines `GeneratorSystem`, which generates a scenario, critiques and refines it, then generates scenario-based questions.
- Extends the SAQ `GeneratorSystem` to reuse its mark-scheme and expected-answer generation methods.
"""
import json
from typing import List, Dict, Any
from openai import OpenAI
from src.llm_response_generation.functions import generate_llm_json_response
from src.data_processing.synthetic.bloom_system_instructions.sbqs import (
    SCENARIO_GENERATION_INSTRUCTION,
    SCENARIO_CRITIQUE_INSTRUCTION,
    SCENARIO_REFINEMENT_INSTRUCTION,
    QUESTION_GEN_MAPPINGS
)
# from src.data_processing.synthetic.generation_and_verification.base_generator_system import BaseGeneratorSystem
from src.data_processing.synthetic.generation_and_verification.saqs.generator import GeneratorSystem as BaseGeneratorSystemSAQ # To inherit generating methods for mark scheme and answer

class GeneratorSystem(BaseGeneratorSystemSAQ):
    """
    Generator system for scenario-based questions (SBQs).
    - Generates a self-contained scenario from course content, then critiques and refines it for compliance.
    - Generates scenario-based questions from the refined scenario.
    - Inherits mark-scheme and expected-answer generation from the SAQ `GeneratorSystem`.
    """

    def __init__(self, client:OpenAI, model_name:str):
        """
        Initialises the SBQ GeneratorSystem with the given OpenAI client and model name.

        Args:
            client (OpenAI): The OpenAI client instance that supports LLM responses for multiple models including Gemini.
            model_name (str): The name of the model to use for generation (must be supported by the endpoint).
        """
        super().__init__(client, model_name)
    
    def generate_scenario(
            self,
            extracted_text:str
        ) -> Dict[str, Any]:
        """
        Generates a realistic, self-contained scenario based on the provided course content.
        Args:
            extracted_text (str): The extracted course content text to base the scenario on.
        Returns:
            Dict[str, Any]: A dictionary containing the generated scenario text.
                            e.g., {
                                    "scenario": "<scenario text>"
                                 }
        """
        scenario_generation_input_json = {
            "course_content": extracted_text
        }
        scenario_generation_prompt_text = f"Generate a JSON object following the schema described in the system message. \n\n" + json.dumps(scenario_generation_input_json)
        scenario_json = generate_llm_json_response(
            client=self.client,
            model_name=self.model_name,
            system_instruction=SCENARIO_GENERATION_INSTRUCTION,
            prompt_text=scenario_generation_prompt_text
        )
        # print(f"Extracted text: {extracted_text}")
        # print(f"Generated scenario JSON: {json.dumps(scenario_json, indent=4)}")
        # print()
        if scenario_json is None:
            return None
        try:
            assert "scenario" in scenario_json, "'scenario' key not found in generated scenario JSON"
            assert isinstance(scenario_json["scenario"], str), "'scenario' value must be a string"
            assert len(scenario_json["scenario"].strip()) > 0, "'scenario' value must be a non-empty string"
        except Exception as e:
            return None
            
        return scenario_json
    
    def critique_generated_scenario(self, scenario_text:str) -> Dict[str, Any]:
        """
        Critiques the generated scenario for compliance with specified goals.
        Args:
            scenario_text (str): The generated scenario text to be critiqued.
        Returns:
            Dict[str, Any]: A dictionary containing the critique of the scenario.
                            e.g.,
                            {
                                "is_compliant": <true|false>,
                                "violations": [
                                    {
                                    "type": "<one of the allowed violation types>",
                                    "span": "<exact offending phrase or sentence copied verbatim from the scenario>",
                                    "reason": "<brief reason this violates the goals>",
                                    "suggested_fix": "<minimal change instruction: delete, or replace with neutral factual equivalent>"
                                    }
                                ],
                                "notes": "<optional short note if needed, otherwise empty string>"
                            }
        
        """
        scenario_critique_input_json = {
            "scenario": scenario_text
        }
        scenario_critique_prompt_text = json.dumps(scenario_critique_input_json)
        critique_json = generate_llm_json_response(
            client=self.client,
            model_name=self.model_name,
            system_instruction=SCENARIO_CRITIQUE_INSTRUCTION,
            prompt_text=scenario_critique_prompt_text
        )
        # print(f"Generated scenario text: {scenario_text}")
        # print(f"Critique JSON: {json.dumps(critique_json, indent=4)}")
        # print()
        try:
            assert "is_compliant" in critique_json, "'is_compliant' key not found in critique JSON"
            assert isinstance(critique_json["is_compliant"], bool), "'is_compliant' value must be a boolean"
            assert "violations" in critique_json, "'violations' key not found in critique JSON"
            assert isinstance(critique_json["violations"], list), "'violations' value must be a list"
            assert "notes" in critique_json, "'notes' key not found in critique JSON"
            assert isinstance(critique_json["notes"], str), "'notes' value must be a string"

            # Check each violation entry
            for violation in critique_json["violations"]:
                assert isinstance(violation, dict), "Each violation must be a dictionary"
                assert "type" in violation, "Each violation must have a 'type' key"
                assert violation["type"] in [
                    "metric_as_conclusion",
                    "teaching_or_definition",
                    "interpretation_or_judgement",
                    "mental_state_or_goal_language",
                    "rationale_or_intent",
                    "recommendation_or_next_step",
                    "question_leakage",
                    "other_rule_violation"
                ], f"Violation type '{violation['type']}' is not recognized"
                assert isinstance(violation["type"] , str), "'type' value in violation must be a string"
                assert "span" in violation, "Each violation must have a 'span' key"
                assert isinstance(violation["span"], str), "'span' value in violation must be a string"
                assert "reason" in violation, "Each violation must have a 'reason' key"
                assert isinstance(violation["reason"], str), "'reason' value in violation must be a string"
                assert "suggested_fix" in violation, "Each violation must have a 'suggested_fix' key"
                assert isinstance(violation["suggested_fix"], str), "'suggested_fix' value in violation must be a string"
        except Exception as e:
            return None
        return critique_json


    def refine_generated_scenario(
            self,
            scenario_text:str,
            critique_json:Dict[str, Any]
            ) -> Dict[str, Any]:
        """
        Refines the generated scenario based on the provided critique.
        - Critique JSON is formatted like so:
        {
            "is_compliant": <true|false>,
            "violations": [
                {
                "type": "<one of the allowed violation types>",
                "span": "<exact offending phrase or sentence copied verbatim from the scenario>",
                "reason": "<brief reason this violates the goals>",
                "suggested_fix": "<minimal change instruction: delete, or replace with neutral factual equivalent>"
                }
            ],
            "notes": "<optional short note if needed, otherwise empty string>"
        }

        Args:
            scenario_text (str): The original generated scenario text.
            critique_json (Dict[str, Any]): The critique of the generated scenario.
        Returns:
            Dict[str, Any]: A dictionary containing the refined scenario text.
                            e.g., {
                                    "scenario": "<refined scenario text>"
                                    }
        """
        refine_input_json = {
            "scenario": scenario_text,
            "critique": critique_json
        }
        refine_prompt_text = json.dumps(refine_input_json)
        refined_scenario_json = generate_llm_json_response(
            client=self.client,
            model_name=self.model_name,
            system_instruction=SCENARIO_REFINEMENT_INSTRUCTION,
            prompt_text=refine_prompt_text
        )
        # print(f"Original scenario text: {scenario_text}")
        # print(f"Critique JSON: {json.dumps(critique_json, indent=4)}")
        # print(f"Refined scenario JSON: {json.dumps(refined_scenario_json, indent=4)}")
        # print()
        try:
            assert "scenario" in refined_scenario_json, "'scenario' key not found in refined scenario JSON"
            assert isinstance(refined_scenario_json["scenario"], str), "'scenario' value must be a string"
            assert len(refined_scenario_json["scenario"].strip()) > 0, "'scenario' value must be a non-empty string"
        except Exception as e:
            return None
        return refined_scenario_json
    
    def generate_sbqs(
            self,
            scenario_text:str,
            system_instruction:str
        ) -> List[Dict[str, Any]]:
        """
        Generates scenario-based questions (SBQs) based on the provided scenario text.

        Args:
            scenario_text (str): The scenario text to base the SBQs on.
            system_instruction (str): The system instruction to guide the SBQ generation.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing a generated SBQ.
                                  e.g., [
                                          {
                                              "question": "<generated question text>",
                                              "bloom_level": "<Bloom's taxonomy level>"
                                          },
                                          ...
                                        ]
        """
        sbq_generation_input_json = {
            "scenario": scenario_text
        }
        sbq_generation_prompt_text = json.dumps(sbq_generation_input_json, indent=4)
        sbq_json = generate_llm_json_response(
            client=self.client,
            model_name=self.model_name,
            system_instruction=system_instruction,
            prompt_text=sbq_generation_prompt_text
        )
        # print(f"Scenario text for SBQ generation: {scenario_text}")
        # print(f"Generated SBQ JSON: {json.dumps(sbq_json, indent=4)}")
        # print()
        try:
            assert isinstance(sbq_json, list), "Generated SBQ JSON must be a list"
            for sbq in sbq_json:
                assert isinstance(sbq, dict), "Each SBQ must be a dictionary"
                assert "question" in sbq, "Each SBQ must have a 'question' key"
                assert isinstance(sbq["question"], str), "'question' value must be a string"
                assert len(sbq["question"].strip()) > 0, "'question' value must be a non-empty string"
                assert "bloom_level" in sbq, "Each SBQ must have a 'bloom_level' key"
                assert isinstance(sbq["bloom_level"], str), "'bloom_level' value must be a string"
                assert sbq["bloom_level"].lower() in QUESTION_GEN_MAPPINGS.keys(), f"'bloom_level' value '{sbq['bloom_level']}' is not recognized"
        except Exception as e:
            return None
        return sbq_json