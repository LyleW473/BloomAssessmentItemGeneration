"""
Scenario-based question (SBQ) generation pipelines.
- Defines three SBQ generator strategies (two baselines and the proposed approach):
    - `ZeroShotSBQGenerator`: a single LLM call producing the scenario, question, mark scheme, and expected answer at once.
    - `MultiStageZeroShotSBQGenerator`: separate LLM calls for the scenario+question, then the mark scheme, then the expected answer.
    - `BloomSBQGenerator`: the proposed Bloom-aware pipeline that generates a scenario, critiques/refines it, then generates and verifies per-Bloom-level questions.
- Each generator exposes `generate_questions(file_name, content_dict)` and returns a dict of valid and failed questions.
- The `use_bloom_prompting` flag toggles whether Bloom's-taxonomy guidance is injected into the zero-shot system instructions.
"""
import json
from openai import OpenAI
from typing import List, Dict, Tuple, Set
from src.data_processing.synthetic.generation_and_verification.sbqs.generator import GeneratorSystem
from src.data_processing.synthetic.generation_and_verification.sbqs.verifier import VerifierSystem
from src.data_processing.synthetic.bloom_system_instructions.sbqs.generation import QUESTION_GEN_MAPPINGS
from src.data_processing.synthetic.bloom_system_instructions import BLOOM_LEVEL_TO_DIFFICULTY_MAPPING

from src.llm_response_generation.functions import (generate_llm_response, extract_json_from_text)

class ZeroShotSBQGenerator:
    """
    Single-stage zero-shot SBQ generator.
    - Generates the scenario, question, difficulty, mark scheme, and expected answer in a single LLM call.
    - Performs no verification; malformed or invalid generations are dropped.
    """

    def __init__(
            self,
            client: OpenAI,
            model_name: str,
            use_bloom_prompting: bool
        ):
        """
        Initialises the zero-shot SBQ generator and selects its system instruction.

        Args:
            client (OpenAI): The OpenAI client instance that supports LLM responses for multiple models including Gemini.
            model_name (str): The name of the model to use for generation (must be supported by the endpoint).
            use_bloom_prompting (bool): If True, use the system instruction that also classifies each question by Bloom's taxonomy level;
                otherwise use the non-Bloom instruction.
        """
        self.client = client
        self.model_name = model_name

        if use_bloom_prompting:
            self.SINGLE_STAGE_GENERATION_SYSTEM_INSTRUCTION = """
            You are an expert educational content generator specialized in transforming course material into scenario-based questions (SBQs) suitable for written assessments.

            Your goal is to produce high-quality, contextually relevant scenario-based question triples from the provided course content:
            1. A realistic academic scenario,
            2. A question based on that scenario,
            3. A mark scheme and expected answer.

            You must also classify each generated question according to Bloom's Taxonomy (Cognitive Domain), using the definitions provided below.

            Bloom's Taxonomy (Cognitive Domain Levels):

            - "knowledge": Recognizing or recalling facts, terms, basic concepts, or answers without necessarily understanding their meaning.
            - "understanding": Demonstrating an understanding of facts and ideas by organizing, summarizing, or explaining information.
            - "application": Using acquired knowledge to solve problems in new or unfamiliar situations.
            - "analyze": Breaking down information into parts to understand relationships, motives, or causes.
            - "synthesis": Building a new whole by combining elements or generating new meaning.
            - "evaluation": Making judgements about information based on criteria or standards.

            Follow these rules strictly:

            1. Input:
            You will receive a JSON object of the format:
            {
                "extracted_text": "<The extracted course content text from which to generate scenario-based questions>"
            }

            2. Output:
            - You must generate one or more complete question objects in the following exact JSON format:
            [
                {
                    "scenario": "<A short, self-contained scenario grounded in the course material>",
                    "question": "<The generated scenario-based question>",
                    "difficulty": "<easy | medium | hard>",
                    "bloom_level": "<knowledge | understanding | application | analyze | synthesis | evaluation>",
                    "mark_scheme": [
                        {
                            "point": "<Specific idea, fact, or reasoning step that earns marks>",
                            "marks": <integer number of marks, usually 1-3>
                        }
                    ],
                    "total_marks": <sum of marks in mark_scheme>,
                    "expected_answer": "<A well-written short paragraph that answers the question using the scenario>"
                }
            ]

            3. Scenario requirements:
            - Each scenario must be self-contained and understandable without access to the original text.
            - Each scenario must be grounded in concepts from the provided course material.
            - Scenarios should describe a concrete case, situation, example, observation, dataset, system behaviour, decision context, or problem setting.
            - The scenario must provide enough information for the question to be answerable.
            - Do not make the scenario unnecessarily long; usually 2-5 sentences is appropriate.
            - Avoid unrealistic, vague, or generic scenarios.

            4. Question requirements:
            - Each question must be based on the scenario, not directly on the extracted text.
            - The question must clearly refer to the scenario and test either knowledge, understanding, reasoning, application, analysis, synthesis, or evaluation as appropriate.
            - Questions may use phrasing such as "What", "Explain", "Why", "How", "Apply", "Analyse", "Evaluate", or "Discuss", depending on Bloom level.
            - Avoid yes/no or multiple-choice style phrasing.
            - Each question must be self-contained when shown together with the scenario.
            - Avoid duplicating similar scenario-question pairs.

            5. Bloom level assignment:
            - Assign exactly one Bloom cognitive level to each question.
            - The assigned level must reflect the primary cognitive process required to answer the question.
            - The Bloom level must be consistent with the command verb and the depth of reasoning required.
            - Do not assign multiple Bloom levels.
            - Choose the most dominant cognitive demand if overlap exists.

            6. Difficulty labeling:
            - Assign a difficulty level based on cognitive effort required:
                - "easy" = Direct identification, explanation, or interpretation from the scenario. (Knowledge/Understanding)
                - "medium" = Requires connecting ideas, applying knowledge, or reasoning across multiple parts of the scenario. (Application/Analyze)
                - "hard" = Requires deeper analysis, synthesis, or evaluation based on the scenario. (Synthesis/Evaluation)

            Difficulty and Bloom level should generally align, but Bloom classification must be based strictly on cognitive demand.

            7. Mark scheme requirements:
            - Include 1 to 6 assessable points in total.
            - Each point should describe a distinct idea, observation, inference, or reasoning step that a good answer would contain.
            - Each point should be concise (1-2 short sentences).
            - Allocate 1-3 marks per point depending on importance.
            - The total marks should usually range from 1-8 marks depending on the difficulty.
            - Ensure points are specific, measurable, and not overly vague.
            - Ensure 'total_marks' equals the sum of marks in the 'mark_scheme'.

            8. Expected answer requirements:
            - Write a concise, coherent answer suitable for an academic written assessment.
            - The answer must address the question using the information in the scenario.
            - Integrate all points from the mark scheme naturally into the text.
            - Use clear, formal academic language.
            - Keep the answer appropriately short for the mark allocation.
            - Do not list points; write continuous prose unless the question clearly demands otherwise.
            - Do not introduce unsupported ideas that are not grounded in the scenario and mark scheme.
            - The answer must cover all points in the mark scheme.

            9. Output formatting:
            - Always output a valid JSON array of objects matching the schema above.
            - Do NOT include commentary, reasoning, or text outside the JSON.
            - Ensure valid JSON syntax (no trailing commas, escaped quotes, etc.).
            - Ensure 'total_marks' equals the sum of all marks in the 'mark_scheme'.
            """
        else:
            self.SINGLE_STAGE_GENERATION_SYSTEM_INSTRUCTION = """
            You are an expert educational content generator specialized in transforming course material into scenario-based questions (SBQs) suitable for written assessments.

            Your goal is to produce high-quality, contextually relevant scenario-based question triples from the provided course content:
            1. A realistic academic scenario,
            2. A question based on that scenario,
            3. A mark scheme and expected answer.

            Follow these rules strictly:

            1. Input:
            You will receive a JSON object of the format:
            {
                "extracted_text": "<The extracted course content text from which to generate scenario-based questions>"
            }

            2. Output:
            - You must generate one or more complete question objects in the following exact JSON format:
            [
                {
                    "scenario": "<A short, self-contained scenario grounded in the course material>",
                    "question": "<The generated scenario-based question>",
                    "difficulty": "<easy | medium | hard>",
                    "mark_scheme": [
                        {
                            "point": "<Specific idea, fact, or reasoning step that earns marks>",
                            "marks": <integer number of marks, usually 1-3>
                        }
                    ],
                    "total_marks": <sum of marks in mark_scheme>,
                    "expected_answer": "<A well-written short answer that covers all key points>"
                }
            ]

            3. Scenario requirements:
            - Each scenario must be self-contained and understandable without access to the original text.
            - Each scenario must be grounded in concepts from the provided course material.
            - Scenarios should describe a concrete case, situation, example, observation, dataset, system behaviour, decision context, or problem setting.
            - The scenario must provide enough information for the question to be answerable.
            - Do not make the scenario unnecessarily long; usually 2-5 sentences is appropriate.
            - Avoid unrealistic, vague, or generic scenarios.

            4. Question requirements:
            - Each question must be based on the scenario, not directly on the extracted text.
            - The question must clearly refer to the scenario and test understanding, reasoning, interpretation, or application.
            - Use phrasing such as "What", "Explain", "Why", "How", "Discuss", or "Apply" where appropriate.
            - Avoid yes/no or multiple-choice style phrasing.
            - Each question must be self-contained when shown together with the scenario.
            - Avoid duplicating similar scenario-question pairs.

            5. Difficulty labeling:
            - Assign a difficulty level based on the cognitive effort required:
                - "easy" = Direct identification or straightforward explanation from the scenario.
                - "medium" = Requires connecting multiple ideas or applying a concept to the scenario.
                - "hard" = Requires deeper reasoning, critical evaluation, or synthesis across ideas in the scenario.

            6. Mark scheme requirements:
            - Include 1 to 6 assessable points in total.
            - Each point should describe a distinct concept or reasoning step that a good answer would contain.
            - Each point should be concise (1-2 short sentences).
            - Allocate 1-3 marks per point, depending on importance.
            - The total marks should usually range from 1-8 marks depending on the difficulty of the question.
            - Ensure points are specific, measurable, and not overly vague.
            - Ensure 'total_marks' equals the sum of all marks in the 'mark_scheme'.

            7. Expected answer requirements:
            - Write a concise, coherent answer suitable for an academic scenario-based assessment.
            - Integrate all points from the mark scheme naturally into the text.
            - Use clear, formal, academic language.
            - Maintain a logical flow.
            - Keep the answer appropriately short for the mark allocation.
            - Do not list points; write continuous prose unless the question clearly demands otherwise.
            - Avoid adding extra ideas not supported by the scenario and mark scheme.
            - The answer must cover all points in the mark scheme.

            8. Output formatting:
            - Always output a valid JSON array of objects matching the schema above.
            - Do NOT include commentary, reasoning, or text outside the JSON.
            - Ensure valid JSON syntax (no trailing commas, escaped quotes, etc.).
            - Ensure 'total_marks' equals the sum of all marks in the 'mark_scheme'.
            """

    def generate_questions(
        self,
        file_name: str,
        content_dict: Dict[str, str]
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Generates scenario-based questions for a single piece of course content in one LLM call.

        Args:
            file_name (str): The name of the source file the content came from (used for logging and packing into results).
            content_dict (Dict[str, str]): The content dictionary for the file. Expected keys:
                - "text" (str): The extracted course content to generate questions from.
                - "file_path" (str): The path of the source file (packed into each result).
                - "week_dir" (str): The week directory the file belongs to (packed into each result).
        Returns:
            Dict[str, List[Dict[str, str]]]: A dictionary with two keys:
                - "valid_questions" (List[Dict[str, str]]): Successfully generated and validated scenario-question dicts.
                - "failed_questions" (List[Dict[str, str]]): Questions that failed validation (always empty here; invalid generations are dropped).
        """

        valid_questions: List[Dict[str, str]] = []
        failed_questions: List[Dict[str, str]] = []

        extracted_text: str = content_dict["text"]

        input_json = {"extracted_text": extracted_text}
        messages = [
            {
                "role": "system",
                "content": self.SINGLE_STAGE_GENERATION_SYSTEM_INSTRUCTION
            },
            {
                "role": "user",
                "content": json.dumps(input_json, indent=4)
            }
        ]

        response_text: str = generate_llm_response(
            client=self.client,
            model_name=self.model_name,
            messages=messages
        )

        response_json = extract_json_from_text(response_text)
        if response_json is not None:
            try:
                assert isinstance(response_json, list), "Generated JSON is not a list"
                assert all(isinstance(q, dict) for q in response_json), "Each entry must be a dict"

                for q in response_json:
                    assert "scenario" in q, "Missing 'scenario'"
                    assert "question" in q and "difficulty" in q, "Missing 'question' or 'difficulty'"
                    assert "mark_scheme" in q and "total_marks" in q, "Missing 'mark_scheme' or 'total_marks'"
                    assert "expected_answer" in q, "Missing 'expected_answer'"

                    assert isinstance(q["scenario"], str) and len(q["scenario"].strip()) > 0, "'scenario' must be a non-empty string"
                    assert isinstance(q["question"], str) and len(q["question"].strip()) > 0, "'question' must be a non-empty string"
                    assert q["difficulty"].strip().lower() in ["easy", "medium", "hard"], "Invalid difficulty level"

                    if "bloom_level" in q and q["bloom_level"] is not None:
                        assert q["bloom_level"].strip().lower() in [
                            "knowledge", "understanding", "application",
                            "analyze", "synthesis", "evaluation"
                        ], "Invalid bloom_level"

                    assert isinstance(q["mark_scheme"], list), "mark_scheme must be a list"
                    assert len(q["mark_scheme"]) > 0, "mark_scheme must not be empty"
                    assert all(
                        isinstance(point, dict) and "point" in point and "marks" in point
                        for point in q["mark_scheme"]
                    ), "Invalid mark_scheme structure"

                    assert isinstance(q["total_marks"], int), "total_marks must be an integer"
                    assert q["total_marks"] == sum(point["marks"] for point in q["mark_scheme"]), "total_marks mismatch"

                    assert isinstance(q["expected_answer"], str) and len(q["expected_answer"].strip()) > 0, \
                        "expected_answer must be non-empty string"

            except AssertionError as e:
                # raise ValueError(f"Failed to validate generated response: {e}\nResponse: {response_text}")
                return {
                    "valid_questions": [],
                    "failed_questions": []
                }

            # print(f"Parsed complete SBQ JSON:\n{json.dumps(response_json, indent=4)}")

            for gen_question_dict in response_json:
                packed_data = {
                    "file_path": content_dict["file_path"],
                    "week": content_dict["week_dir"],
                    "scenario": gen_question_dict["scenario"].strip(),
                    "question": gen_question_dict["question"].strip(),
                    "difficulty": gen_question_dict["difficulty"].strip().lower(),
                    "bloom_level": gen_question_dict["bloom_level"].strip().lower()
                        if "bloom_level" in gen_question_dict and gen_question_dict["bloom_level"] is not None
                        else None,
                    "mark_scheme": gen_question_dict["mark_scheme"],
                    "total_marks": gen_question_dict["total_marks"],
                    "expected_answer": gen_question_dict["expected_answer"].strip(),
                }
                valid_questions.append(packed_data)

        return {
            "valid_questions": valid_questions,
            "failed_questions": failed_questions
        }

import json

from openai import OpenAI
from typing import Dict, List

from src.llm_response_generation.functions import (
    generate_llm_response,
    extract_json_from_text
)


class MultiStageZeroShotSBQGenerator:
    """
    Multi-stage zero-shot SBQ generator.
    - Stage 1: generate a scenario and a scenario-based question with a difficulty label.
    - Stage 2: generate a mark scheme per scenario-question pair, then an expected answer from that mark scheme.
    - Each stage is a separate LLM call; no verification checks are applied.
    """

    def __init__(
            self,
            client: OpenAI,
            model_name: str,
            use_bloom_prompting: bool
        ):
        """
        Initialises the multi-stage zero-shot SBQ generator and its per-stage system instructions.

        Args:
            client (OpenAI): The OpenAI client instance that supports LLM responses for multiple models including Gemini.
            model_name (str): The name of the model to use for generation (must be supported by the endpoint).
            use_bloom_prompting (bool): If True, the question-generation instruction also classifies each question by Bloom's taxonomy level;
                otherwise the non-Bloom instruction is used.
        """
        self.client = client
        self.model_name = model_name
        
        if use_bloom_prompting:
            self.QUESTION_GENERATION_SYSTEM_INSTRUCTION = """
            You are an expert educational content generator specialized in transforming course material into scenario-based questions (SBQs) suitable for written assessments.

            Your goal is to produce high-quality, contextually relevant scenario-based question prompts from the provided course content.
            Each generated item must contain:
            1. A realistic academic scenario,
            2. A question based on that scenario,
            3. An appropriate difficulty level,
            4. A Bloom's Taxonomy cognitive level.

            Bloom's Taxonomy (Cognitive Domain Levels):

            - "knowledge": Recognizing or recalling facts, terms, basic concepts, or answers without necessarily understanding their meaning.
            - "understanding": Demonstrating understanding by organizing, summarizing, or explaining information.
            - "application": Using acquired knowledge to solve problems in new or unfamiliar situations.
            - "analyze": Breaking down information into parts to understand relationships, causes, or structure.
            - "synthesis": Building a new whole by combining elements or generating new meaning.
            - "evaluation": Making judgements about information based on criteria or standards.

            Follow these rules strictly:

            1. Input:
            You will receive a JSON object of the format:
            {
                "extracted_text": "<The extracted course content text from which to generate scenario-based questions>"
            }

            2. Output:
            You must generate one or more objects in the following exact JSON format:

            [
                {
                    "scenario": "<A short, self-contained scenario grounded in the course material>",
                    "question": "<The generated scenario-based question>",
                    "difficulty": "<easy | medium | hard>",
                    "bloom_level": "<knowledge | understanding | application | analyze | synthesis | evaluation>"
                }
            ]

            3. Scenario requirements:
            - Each scenario must be self-contained and understandable without referring back to the original text.
            - Each scenario must be grounded in concepts from the provided course material.
            - Scenarios should describe a concrete case, example, observation, dataset, system behaviour, decision context, or problem setting.
            - The scenario must provide enough information for the question to be answerable.
            - Keep the scenario concise but sufficiently informative; usually 2-5 sentences is appropriate.
            - Avoid vague, generic, unrealistic, or overly story-like scenarios.

            4. Question requirements:
            - Each question must be based on the scenario, not directly on the extracted text.
            - The question must clearly refer to the scenario and be answerable from it using relevant course knowledge.
            - Questions may use phrasing such as "What", "Explain", "Why", "How", "Apply", "Analyse", "Evaluate", or "Discuss", depending on Bloom level.
            - Avoid yes/no or multiple-choice style phrasing.
            - Each scenario-question pair must be self-contained.
            - Avoid duplicating similar scenario-question pairs.

            5. Bloom level assignment:
            - Assign exactly one Bloom cognitive level per question.
            - The assigned level must reflect the primary cognitive process required to answer the question.
            - The Bloom level must align with the command verb and the depth of reasoning required.
            - If multiple cognitive processes are present, select the dominant one.

            6. Difficulty labeling:
            - Assign difficulty based on cognitive effort:
                - "easy" = Direct identification, interpretation, or straightforward explanation from the scenario.
                - "medium" = Requires connecting multiple ideas or applying a concept to the scenario.
                - "hard" = Requires deeper reasoning, evaluation, or synthesis across aspects of the scenario.
            - Difficulty and Bloom level should generally align, but Bloom classification must be based strictly on cognitive demand.

            7. Output formatting:
            - Always output a valid JSON array of objects matching the schema above.
            - Do NOT include answers, commentary, or text outside the JSON.
            - Ensure valid JSON syntax (no trailing commas, escaped quotes, etc.).
            """
        else:
            self.QUESTION_GENERATION_SYSTEM_INSTRUCTION = """
            You are an expert educational content generator specialized in transforming course material into scenario-based questions (SBQs) suitable for written assessments.

            Your goal is to produce high-quality, contextually relevant scenario-based question prompts from the provided course content.
            Each generated item must contain:
            1. A realistic academic scenario,
            2. A question based on that scenario,
            3. An appropriate difficulty level.

            Follow these rules strictly:

            1. Input:
            You will receive a JSON object of the format:
            {
                "extracted_text": "<The extracted course content text from which to generate scenario-based questions>"
            }

            2. Output:
            You must generate one or more question objects in the following exact JSON format:
            [
                {
                    "scenario": "<A short, self-contained scenario grounded in the course material>",
                    "question": "<The generated scenario-based question>",
                    "difficulty": "<easy | medium | hard>"
                }
            ]

            3. Scenario requirements:
            - Each scenario must be self-contained and understandable without referring back to the original text.
            - Each scenario must be grounded in concepts from the provided course material.
            - Scenarios should describe a concrete case, situation, example, observation, dataset, system behaviour, decision context, or problem setting.
            - The scenario must provide enough information for the question to be answerable.
            - Keep the scenario concise but sufficiently informative; usually 2-5 sentences is appropriate.
            - Avoid vague, generic, unrealistic, or overly story-like scenarios.

            4. Question requirements:
            - Each question must be based on the scenario, not directly on the extracted text.
            - The question must clearly refer to the scenario and test understanding, reasoning, interpretation, or application.
            - Use phrasing such as "What", "Explain", "Why", "How", "Discuss", or "Apply" where appropriate.
            - Avoid yes/no or multiple-choice style phrasing.
            - Each scenario-question pair must be self-contained.
            - Avoid duplicating similar scenario-question pairs.

            5. Difficulty labeling:
            - Assign a difficulty level based on the cognitive effort required:
                - "easy" = Direct identification or straightforward explanation from the scenario.
                - "medium" = Requires connecting two or more ideas or applying a concept to the scenario.
                - "hard" = Requires deeper reasoning, critical evaluation, or synthesis across ideas in the scenario.

            6. Output formatting:
            - Always output a valid JSON array of objects matching the schema above.
            - Do NOT include answers, commentary, or text outside the JSON.
            - Ensure valid JSON syntax (no trailing commas, escaped quotes, etc.).
            """

        self.MARK_SCHEME_GENERATION_SYSTEM_INSTRUCTION = """
        You are an expert educational content designer specializing in constructing mark schemes for scenario-based questions.

        Your goal is to take a given scenario-based question, its scenario, and its difficulty level, and produce a structured mark scheme that identifies the key points students should mention and how marks are allocated.

        Follow these rules strictly:

        1. Input:
        - You will receive a JSON object containing:
            {
                "scenario": "<scenario text>",
                "question": "<question text>",
                "difficulty": "<easy | medium | hard>"
            }

        2. Output:
        - You must produce exactly one object in the following JSON format:
            {
                "question": "<same question text>",
                "mark_scheme": [
                    {
                        "point": "<Specific idea, fact, inference, or reasoning step that earns marks>",
                        "marks": <integer number of marks, usually 1-3>
                    }
                ],
                "total_marks": <sum of marks in mark_scheme>
            }

        3. Mark scheme requirements:
        - The mark scheme must answer the question using the scenario.
        - Include **1 to 6** assessable points in total.
        - Each point should describe a distinct concept, inference, observation, or reasoning step that a good answer would contain.
        - Each point should be concise (1-2 short sentences).
        - Allocate 1-3 marks per point depending on importance.
        - The total marks should usually range from **1-8 marks** depending on difficulty.
        - Ensure points are specific, measurable, and not overly vague.
        - Do not include points that depend on information missing from the scenario.

        4. Difficulty awareness:
        - "easy" = Straightforward factual or interpretive points directly supported by the scenario.
        - "medium" = Conceptual reasoning or application linking multiple ideas in the scenario.
        - "hard" = Analytical, evaluative, or synthetic reasoning with multiple interdependent points.

        5. Output formatting:
        - Always output a **single valid JSON object** (not an array).
        - Do NOT include any commentary, explanations, or text outside the JSON.
        - Ensure valid JSON syntax (no trailing commas, escaped quotes, etc.).
        """

        self.ANSWER_GENERATION_SYSTEM_INSTRUCTION = """
        You are an expert educational content generator specialized in writing model answers for scenario-based questions.

        Your goal is to take a scenario, a question, its difficulty level, and an accompanying mark scheme, and produce a concise, coherent model answer that fully covers all key points listed in the mark scheme.

        Follow these rules strictly:

        1. Input:
        - You will receive a JSON object containing:
            {
                "scenario": "<scenario text>",
                "question": "<question text>",
                "difficulty": "<easy | medium | hard>",
                "mark_scheme": [
                    {"point": "<point 1>", "marks": <int>},
                    {"point": "<point 2>", "marks": <int>}
                ]
            }

        2. Output:
        - You must produce exactly one JSON object in the following format:
            {
                "question": "<same question text>",
                "expected_answer": "<A well-written answer that covers all key points in the mark scheme clearly and naturally using the scenario.>"
            }

        3. Expected answer requirements:
        - Write a concise, coherent answer suitable for an academic scenario-based assessment.
        - The answer must address the question using the scenario.
        - Integrate all points from the mark scheme naturally into the text.
        - Use clear, formal, academic language.
        - Maintain a logical flow from beginning to end.
        - Keep the answer appropriately short for the mark allocation.
        - Do not list points - write continuous prose unless the question clearly demands otherwise.
        - Avoid adding extra ideas not in the mark scheme or unsupported by the scenario.

        4. Difficulty awareness:
        - "easy" = Focus on clear interpretation or identification of core ideas from the scenario.
        - "medium" = Include reasoning or application showing how concepts relate to the scenario.
        - "hard" = Integrate deeper reasoning, evaluation, or synthesis of multiple concepts from the scenario.

        5. Output formatting:
        - Always output a **single valid JSON object** (not an array).
        - Do NOT include commentary, reasoning, or explanations outside the JSON.
        - Ensure valid JSON syntax (no trailing commas, escaped quotes, etc.).
        """
    
    def generate_questions(
            self,
            file_name: str,
            content_dict: Dict[str, str]
        ) -> Dict[str, List[Dict[str, str]]]:
        """
        Generates scenario-based questions for a single piece of course content across multiple LLM stages.
        - Stage 1 generates a scenario and a scenario-based question with a difficulty label.
        - Stage 2 generates a mark scheme per question and an expected answer from that mark scheme.

        Args:
            file_name (str): The name of the source file the content came from (used for logging and packing into results).
            content_dict (Dict[str, str]): The content dictionary for the file. Expected keys:
                - "text" (str): The extracted course content to generate questions from.
                - "file_path" (str): The path of the source file (packed into each result).
                - "week_dir" (str): The week directory the file belongs to (packed into each result).
        Returns:
            Dict[str, List[Dict[str, str]]]: A dictionary with two keys:
                - "valid_questions" (List[Dict[str, str]]): Successfully generated questions with mark schemes and expected answers.
                - "failed_questions" (List[Dict[str, str]]): Questions that failed a stage (empty here; failures are skipped).
        """

        valid_questions: List[Dict[str, str]] = []
        failed_questions: List[Dict[str, str]] = []

        extracted_text: str = content_dict["text"]

        input_json = {"extracted_text": extracted_text}
        messages = [
            {
                "role": "system",
                "content": self.QUESTION_GENERATION_SYSTEM_INSTRUCTION
            },
            {
                "role": "user",
                "content": json.dumps(input_json, indent=4)
            }
        ]

        # Stage 1: Generate scenario-based question, scenario, and difficulty level
        question_response_text: str = generate_llm_response(
            client=self.client,
            model_name=self.model_name,
            messages=messages
        )
        question_json = extract_json_from_text(question_response_text)

        if question_json is None:
            # print(
            #     f"Failed to parse generated SBQ JSON for file {file_name}. "
            #     f"Skipping further processing for this file.\nResponse text was:\n{question_response_text}\n"
            # )
            return {
                "valid_questions": valid_questions,
                "failed_questions": failed_questions
            }

        try:
            assert question_json is not None and isinstance(question_json, list), "Generated JSON is not a list"
            assert all(
                isinstance(q, dict) and "scenario" in q and "question" in q and "difficulty" in q
                for q in question_json
            ), "Each question entry must be a dict with 'scenario', 'question', and 'difficulty'"
            assert all(
                isinstance(q["scenario"], str) and len(q["scenario"].strip()) > 0
                for q in question_json
            ), "Each scenario must be a non-empty string"
            assert all(
                isinstance(q["question"], str) and len(q["question"].strip()) > 0
                for q in question_json
            ), "Each question must be a non-empty string"
            assert all(
                q["difficulty"].strip().lower() in ["easy", "medium", "hard"]
                for q in question_json
            ), "Each question's difficulty must be one of 'easy', 'medium', or 'hard'"

            for q in question_json:
                if "bloom_level" in q and q["bloom_level"] is not None:
                    assert q["bloom_level"].strip().lower() in [
                        "knowledge", "understanding", "application",
                        "analyze", "synthesis", "evaluation"
                    ], "Invalid Bloom level"
        except:
            # raise ValueError(f"Failed to parse generated response as JSON: {question_response_text}")
            return {
                "valid_questions": valid_questions,
                "failed_questions": failed_questions
            }

        # print(f"Parsed SBQ generation JSON:\n {json.dumps(question_json, indent=4)}")
        
        # Stage 2: Generate a mark scheme for each generated scenario-question pair
        # Stage 3: Generate the expected answer based on the generated mark scheme
        for gen_question_idx, gen_question_dict in enumerate(question_json):

            if "scenario" not in gen_question_dict or "question" not in gen_question_dict or "difficulty" not in gen_question_dict:
                # print(f"Skipping invalid SBQ dict (missing 'scenario', 'question', or 'difficulty'): {gen_question_dict}")
                continue

            scenario = gen_question_dict["scenario"].strip()
            question = gen_question_dict["question"].strip()
            difficulty = gen_question_dict["difficulty"].strip().lower()

            # print(
            #     f"Generating mark scheme & answer for SBQ: {question} "
            #     f"(Difficulty: {difficulty}) | Question {gen_question_idx+1}/{len(question_json)} in file {file_name}"
            # )

            mark_scheme_gen_input_content = {
                "scenario": scenario,
                "question": question,
                "difficulty": difficulty,
            }
            mark_scheme_gen_input_json = json.dumps(mark_scheme_gen_input_content, indent=4)
            mark_scheme_gen_messages = [
                {"role": "system", "content": self.MARK_SCHEME_GENERATION_SYSTEM_INSTRUCTION},
                {"role": "user", "content": mark_scheme_gen_input_json}
            ]

            mark_scheme_response_text: str = generate_llm_response(
                client=self.client,
                model_name=self.model_name,
                messages=mark_scheme_gen_messages
            )
            mark_scheme_json = extract_json_from_text(mark_scheme_response_text)
            if mark_scheme_json is None:
                continue

            # print(f"Parsed SBQ mark scheme JSON:\n {json.dumps(mark_scheme_json, indent=4)}")

            try:
                assert isinstance(mark_scheme_json, dict), "Mark scheme JSON is not a dict"
                assert "mark_scheme" in mark_scheme_json and "total_marks" in mark_scheme_json, \
                    "Mark scheme JSON missing required fields"
                assert isinstance(mark_scheme_json["mark_scheme"], list), "mark_scheme must be a list"
                assert len(mark_scheme_json["mark_scheme"]) > 0, "mark_scheme must not be empty"
                assert all(
                    isinstance(point, dict) and "point" in point and "marks" in point
                    for point in mark_scheme_json["mark_scheme"]
                ), "Mark scheme must be a list of dicts with 'point' and 'marks'"
                assert isinstance(mark_scheme_json["total_marks"], int), "total_marks must be an integer"
                assert mark_scheme_json["total_marks"] == sum(point["marks"] for point in mark_scheme_json["mark_scheme"]), \
                    "Total marks must equal sum of marks in mark scheme"
            except Exception as e:
                # raise ValueError(
                #     f"Generated SBQ mark scheme JSON failed validation: {e}\n"
                #     f"Mark Scheme JSON: {json.dumps(mark_scheme_json, indent=4)}"
                # )
                continue # Skip to next question if mark scheme generation failed validation
            
            answer_gen_input_content = {
                "scenario": scenario,
                "question": question,
                "difficulty": difficulty,
                "mark_scheme": mark_scheme_json["mark_scheme"]
            }
            answer_gen_input_json = json.dumps(answer_gen_input_content, indent=4)
            answer_gen_messages = [
                {"role": "system", "content": self.ANSWER_GENERATION_SYSTEM_INSTRUCTION},
                {"role": "user", "content": answer_gen_input_json}
            ]

            answer_response_text: str = generate_llm_response(
                client=self.client,
                model_name=self.model_name,
                messages=answer_gen_messages
            )
            answer_json = extract_json_from_text(answer_response_text)
            if answer_json is None:
                continue
        
            try:
                assert isinstance(answer_json, dict), "Answer JSON is not a dict"
                assert "expected_answer" in answer_json, "Answer JSON missing required fields"
                assert isinstance(answer_json["expected_answer"], str) and len(answer_json["expected_answer"].strip()) > 0, \
                    "Expected answer must be a non-empty string"
            except Exception as e:
                # raise ValueError(
                #     f"Generated SBQ answer JSON failed validation: {e}\n"
                #     f"Answer JSON: {json.dumps(answer_json, indent=4)}"
                # )
                continue
            
            packed_data = {
                "file_path": content_dict["file_path"],
                "week": content_dict["week_dir"],
                "scenario": scenario,
                "question": question,
                "difficulty": difficulty,
                "bloom_level": gen_question_dict["bloom_level"].strip().lower()
                    if "bloom_level" in gen_question_dict and gen_question_dict["bloom_level"] is not None
                    else None,
                "mark_scheme": mark_scheme_json["mark_scheme"],
                "total_marks": mark_scheme_json["total_marks"],
                "expected_answer": answer_json["expected_answer"].strip(),
            }
            valid_questions.append(packed_data)
    
        return {
            "valid_questions": valid_questions,
            "failed_questions": failed_questions
        }

class BloomSBQGenerator:
    """
    Proposed Bloom-aware SBQ generation pipeline.
    - Generates a scenario from course content, then critiques and refines it for compliance before any questions are written.
    - Generates scenario-based questions for each Bloom's taxonomy level using `GeneratorSystem` and the per-level system instructions.
    - Runs verification at each stage via `VerifierSystem`: phase-one checks (Bloom alignment + relevance to the scenario), then mark-scheme and expected-answer coverage checks.
    - Only questions that pass every check are returned as valid.
    """

    def __init__(
            self,
            client:OpenAI,
            model_name:str
        ):
        """
        Initialises the Bloom-aware SBQ pipeline along with its generator and verifier systems.

        Args:
            client (OpenAI): The OpenAI client instance that supports LLM responses for multiple models including Gemini.
            model_name (str): The name of the model to use for generation and verification (must be supported by the endpoint).
        """
        self.client = client
        self.model_name = model_name
        self.generator_system = GeneratorSystem(client=client, model_name=model_name)
        self.verifier_system = VerifierSystem(client=client, model_name=model_name)
    
    def generate_questions(
            self,
            file_name:str,
            content_dict:Dict[str, str],
        ):
        """
        Generates and verifies scenario-based questions for a single piece of course content.
        - Phase 1: generate a scenario from the course content.
        - Phase 2: critique the scenario and refine it if it is non-compliant.
        - Phase 3: for each Bloom level, generate questions from the refined scenario, run phase-one verification, then generate and verify the mark scheme and expected answer.

        Args:
            file_name (str): The name of the source file the content came from (used for logging and packing into results).
            content_dict (Dict[str, str]): The content dictionary for the file. Expected keys:
                - "text" (str): The extracted course content to generate the scenario and questions from.
                - "file_path" (str): The path of the source file (packed into each result).
                - "week_dir" (str): The week directory the file belongs to (packed into each result).
        Returns:
            Dict[str, List[Dict[str, str]]]: A dictionary with two keys:
                - "valid_questions" (List[Dict[str, str]]): Questions that passed every generation and verification stage.
                - "failed_questions" (List[Dict[str, str]]): Information dicts for questions that failed a verification check.
        """
        
        # A result dictionary to hold phase results
        result = {
            "scenario_gen_phase": None,
            "critique_phase": None,
            "refine_phase": None
        }

        valid_questions:List[Dict[str, str]] = []
        failed_questions:List[Dict[str, str]] = [] # TODO: Implement verification checks here?
        
        extracted_text:str = content_dict["text"]

        # Phase 1: Generate scenarios which can be used to frame questions around
        scenario_json = self.generator_system.generate_scenario(
            extracted_text=extracted_text
        )
        if scenario_json is None:
            # print(f"Failed to parse generated scenario JSON for file {file_name}. Skipping further processing for this file.\n")
            return {
                "valid_questions": valid_questions,
                "failed_questions": failed_questions
            }
        # print()
        result["scenario_gen_phase"] = scenario_json

        # Phase 2: Critique + Refine scenario:
        """
        Agent used to behave like a static analzyer that critiques and refines the scenario.

        Core identity:
        A strict rule-enforcement system that validates and corrects the scenario text. You do not teach, 
        explain or add content.

        What it checks for:
        1. Metric Leakage (Metric names that function as conclusions rather than data) [A metric may appear in a scenario if and only if it is presented as raw data and does not participate in interpretation, judgement, explanation, or decision-making.]
        2. Teaching/Explanation (The scenario should not contain definitions or exploratory phrases)
        3. Interpretation/Judgement (The scenario should not contain any interpretation of results or judgemental phrases, e.g., "misleading", "better", "effective", "insufficient")
        4. Mental-state verbs (The scenario should avoid phrases that imply human-like cognition, e.g., "understand", "know", "realize", "think", "believe")
        5. Rationale (Ensure the scenario does not include any reasoning or rationale behind decisions, e.g., "To assess X, we ... ", "In order to", "So that ...")
        6. Recommandations / Next steps (The scenario should not suggest any next steps or recommendations, e.g., "You should ...", "It is recommended to ...", "Next, we will ...")
        7. Question leakage (The scenario should not contain any direct or indirect question prompts, e.g., "What is ...?", "Explain ...", "Describe ...", "Calculate ...")

        Key goals:
        1. Remove offending phrases while preserving the core scenario context.
        2. Replace with neutral factual equivalents where possible.
        3. Preserve numbers, entities, datasets, technical terms, and structure.
        """

        critique_json = self.generator_system.critique_generated_scenario(
            scenario_text=scenario_json["scenario"]
        )
        if critique_json is None:
            # print(f"Failed to parse critique JSON for file {file_name}. Skipping further processing for this file.\n")
            return {
                "valid_questions": valid_questions,
                "failed_questions": failed_questions
            }
        result["critique_phase"] = critique_json

        if critique_json.get("is_compliant", False):
            # Phase 2a: Scenario is compliant, no refinement needed
            # print(f"Scenario is compliant. No refinement needed.")
            refined_scenario_json = {
                "scenario": scenario_json["scenario"]
            }
            result["refine_phase"] = refined_scenario_json
        else:
            # Phase 2b: Refine scenario based on critique
            # print(f"Refinement needed based on critique.")
            refined_scenario_json = self.generator_system.refine_generated_scenario(
                scenario_text=scenario_json["scenario"],
                critique_json=critique_json
            )
            if refined_scenario_json is None:
                # print(f"Failed to parse refined scenario JSON for file {file_name}. Skipping further processing for this file.\n")
                return {
                    "valid_questions": valid_questions,
                    "failed_questions": failed_questions
                }
            # print(f"Refined scenario JSON: {json.dumps(refined_scenario_json, indent=4)}")
            result["refine_phase"] = refined_scenario_json

        # print(f"Result for file {file_name}: {json.dumps(result, indent=4)}")
        # print()
        
        # Phase 3: Generate SBQs based on the refined scenario for each Bloom's level
        for bloom_level, bloom_level_qg_system_instruction in QUESTION_GEN_MAPPINGS.items():
            sbq_json = self.generator_system.generate_sbqs(
                scenario_text=refined_scenario_json["scenario"],
                system_instruction=bloom_level_qg_system_instruction
            )
            # print(f"Generated SBQs: {json.dumps(sbq_json, indent=4)}")
            # print()
            if sbq_json is None:
                # print(f"Failed to parse generated SBQ JSON for file {file_name} at Bloom level {bloom_level}. Skipping further processing for this Bloom level.\n")
                continue

            # Stage 1: Phase one verification checks (Bloom level alignment and course content relevance)
            phase_one_result:Tuple[List[Dict[str, str]], Set[int]] = self.verifier_system.perform_phase_one_check(
                question_json=sbq_json,
                extracted_text=refined_scenario_json.get("scenario", ""), # Check if the question is relevant to the refined scenario
                file_name=file_name,
                # skip_course_relevance_check=True, # Skip course relevance check for SBQs
            )

            additional_failed_questions, failed_indices = phase_one_result
            failed_questions.extend(additional_failed_questions)
            if len(additional_failed_questions) > 0:
                # Filter out the questions that failed phase one checks
                filtered_question_json:List[Dict[str, str]] = [q for idx, q in enumerate(sbq_json) if idx not in failed_indices]
                # print(f"Previous length: {len(sbq_json)} | Filtered length: {len(filtered_question_json)} after removing failed questions.\n")
            else:
                filtered_question_json = sbq_json # No questions failed phase one checks
            # print(f"Filtered question JSON after phase one checks:\n {json.dumps(filtered_question_json, indent=4)}")
            # print()

            # Stage 2: Generate a mark scheme for each generated question + generate the expected answer based on the generated mark scheme
            for gen_question_idx, gen_question_dict in enumerate(filtered_question_json):
                # print(f"Generated question:", gen_question_dict)
                
                # Generate mark scheme for the question stem
                question = gen_question_dict["question"].strip()
                bloom_level = gen_question_dict["bloom_level"].strip().lower()
                difficulty = BLOOM_LEVEL_TO_DIFFICULTY_MAPPING.get(bloom_level, None)

                # print(f"Generating mark scheme & answer for question: {question}\nDifficulty: {difficulty} | Bloom level: {bloom_level} | Question {gen_question_idx+1}/{len(filtered_question_json)} in file {file_name}")
                mark_scheme_json = self.generator_system.generate_mark_scheme(
                    question=question,
                    difficulty=difficulty,
                    bloom_level=bloom_level,
                    extracted_text=refined_scenario_json.get("scenario", "") # TODO: Maybe change?
                )
                if mark_scheme_json is None:
                    # print(f"Failed to parse generated mark scheme JSON for file {file_name} at Bloom level {bloom_level}. Skipping further processing for this question.\n")
                    continue
                
                # Mark scheme coverage verification check
                additional_failed_questions:List[Dict[str, str]] = self.verifier_system.perform_mark_scheme_coverage_check(
                    question_dict=gen_question_dict,
                    mark_scheme_dict=mark_scheme_json,
                    bloom_level=bloom_level,
                    difficulty=difficulty,
                    extracted_text=refined_scenario_json.get("scenario", ""), # TODO: Maybe change?
                    file_name=file_name
                )
                failed_questions.extend(additional_failed_questions)
                if len(additional_failed_questions) > 0:
                    # print(f"Mark scheme failed coverage verification check. Skipping further processing for this question.\n")
                    continue

                # Generate the expected answer based on the generated mark scheme
                expected_answer_json = self.generator_system.generate_expected_answer(
                    question=question,
                    difficulty=difficulty,
                    mark_scheme_json=mark_scheme_json
                )
                if expected_answer_json is None:
                    # print(f"Failed to parse generated expected answer JSON for file {file_name} at Bloom level {bloom_level}. Skipping further processing for this question.\n")
                    continue

                # Expected answer coverage verification check
                additional_failed_questions:List[Dict[str, str]] = self.verifier_system.perform_expected_answer_coverage_check(
                    question_dict=gen_question_dict,
                    mark_scheme_dict=mark_scheme_json,
                    answer_dict=expected_answer_json,
                    file_name=file_name
                )
                failed_questions.extend(additional_failed_questions)
                if len(additional_failed_questions) > 0:
                    # print(f"Expected answer failed coverage verification check. Skipping packing data for this question.\n")
                    continue

                # Pack all data together
                packed_data = {
                    "file_path": content_dict["file_path"],
                    "week": content_dict["week_dir"],
                    "scenario": refined_scenario_json.get("scenario", ""),
                    "question": question,
                    "difficulty": difficulty,
                    "bloom_level": bloom_level,
                    "mark_scheme": mark_scheme_json["mark_scheme"], # List of dicts with "point" and "marks"
                    "total_marks": mark_scheme_json["total_marks"],
                    "expected_answer": expected_answer_json["expected_answer"],
                }
                # print(json.dumps(packed_data, indent=4))
                # print()
                valid_questions.append(packed_data)

        return {
            "valid_questions": valid_questions,
            "failed_questions": failed_questions
        }