"""
Short-answer question (SAQ) generation pipelines.
- Defines three SAQ generator strategies (two baselines and the proposed approach):
    - `ZeroShotSAQGenerator`: a single LLM call producing the question, mark scheme, and expected answer at once.
    - `MultiStageZeroShotSAQGenerator`: separate LLM calls for the question, then the mark scheme, then the expected answer.
    - `BloomSAQGenerator`: the proposed Bloom-aware pipeline that generates per-Bloom-level questions and verifies each generation stage.
- Each generator exposes `generate_questions(file_name, content_dict)` and returns a dict of valid and failed questions.
- The `use_bloom_prompting` flag toggles whether Bloom's-taxonomy guidance is injected into the zero-shot system instructions.
"""
import json
import os

from openai import OpenAI
from typing import Dict, List, Tuple, Set
from src.data_processing.synthetic.generation_and_verification.saqs.generator import GeneratorSystem
from src.data_processing.synthetic.generation_and_verification.saqs.verifier import VerifierSystem
from src.data_processing.synthetic.bloom_system_instructions.saqs import QUESTION_GEN_MAPPINGS
from src.data_processing.synthetic.bloom_system_instructions.shared import BLOOM_LEVEL_TO_DIFFICULTY_MAPPING

from src.llm_response_generation.functions import (generate_llm_response, extract_json_from_text)

class ZeroShotSAQGenerator:
    """
    Single-stage zero-shot SAQ generator.
    - Generates the question, difficulty, mark scheme, and expected answer in a single LLM call.
    - Performs no verification; malformed or invalid generations are dropped.
    """

    def __init__(
            self,
            client:OpenAI,
            model_name:str,
            use_bloom_prompting:bool
        ):
        """
        Initialises the zero-shot SAQ generator and selects its system instruction.

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
            You are an expert educational content generator specialized in transforming course material into short-answer questions suitable for written assessments.

            Your goal is to produce high-quality, contextually relevant short-answer questions with mark schemes and model answers from the provided course content.

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
                "extracted_text": "<The extracted course content text from which to generate questions>"
            }

            2. Output:
            - You must generate one or more complete question objects in the following exact JSON format:
            [
                {
                    "question": "<The generated short-answer question>",
                    "difficulty": "<easy | medium | hard>",
                    "bloom_level": "<knowledge | understanding | application | analyze | synthesis | evaluation>",
                    "mark_scheme": [
                        {
                            "point": "<Specific idea, fact, or reasoning step that earns marks>",
                            "marks": <integer number of marks, usually 1-3>
                        }
                    ],
                    "total_marks": <sum of marks in mark_scheme>,
                    "expected_answer": "<A well-written short paragraph (50-100 words) that covers all key points>"
                }
            ]

            3. Question requirements:
            - Each question must elicit a short, paragraph-style written answer (approximately 50-100 words).
            - Questions should test understanding, explanation, reasoning, or conceptual comparison rather than simple recall (unless explicitly appropriate).
            - Use phrasing such as "Explain", "Describe", "Compare", "Discuss", "Evaluate", or "Why".
            - Avoid yes/no or multiple-choice style phrasing.
            - Each question must be self-contained and understandable without referring back to the original text.
            - Avoid duplicating similar questions.

            4. Bloom level assignment:
            - Assign exactly one Bloom cognitive level to each question.
            - The assigned level must reflect the primary cognitive process required to answer the question.
            - The Bloom level must be consistent with the command verb and the depth of reasoning required.
            - Do not assign multiple Bloom levels.
            - Choose the most dominant cognitive demand if overlap exists.

            5. Difficulty labeling:
            - Assign a difficulty level based on cognitive effort required:
                - "easy" = Explains or defines a single concept directly from the material (typically remember/understand).
                - "medium" = Requires connecting multiple ideas or applying a concept (often apply/analyze).
                - "hard" = Requires deeper reasoning, evaluation, or synthesis (often evaluate/create).

            Difficulty and Bloom level should generally align, but Bloom classification must be based strictly on cognitive demand.

            6. Mark scheme requirements:
            - Include 4 to 8 assessable points in total.
            - Each point should describe a distinct concept or reasoning step that a good answer would contain.
            - Each point should be concise (1-2 short sentences).
            - Allocate 1-3 marks per point depending on importance.
            - The total marks should usually range from 3-10 marks depending on the difficulty.
            - Ensure points are specific, measurable, and not overly vague.
            - Ensure total_marks equals the sum of marks in the mark_scheme.

            7. Expected answer requirements:
            - Write a concise, coherent paragraph suitable for an academic short-answer assessment.
            - Integrate all points from the mark scheme naturally into the text.
            - Use clear, formal academic language.
            - Maintain logical flow from introduction to conclusion.
            - Keep length between 50 and 100 words.
            - Do not list points; write continuous prose.
            - Do not introduce ideas not present in the mark scheme.
            - The answer must cover all points in the mark scheme.

            8. Output formatting:
            - Always output a valid JSON array of objects matching the schema above.
            - Do NOT include commentary, reasoning, or text outside the JSON.
            - Ensure valid JSON syntax (no trailing commas, escaped quotes, etc.).
            - Ensure total_marks equals the sum of all marks in the mark_scheme.
            """
        else:
            self.SINGLE_STAGE_GENERATION_SYSTEM_INSTRUCTION = """
            You are an expert educational content generator specialized in transforming course material into short-answer questions suitable for written assessments.

            Your goal is to produce high-quality, contextually relevant short-answer questions with mark schemes and model answers from the provided course content.

            Follow these rules strictly:

            1. Input:
            You will receive a JSON object of the format:
            {
                "extracted_text": "<The extracted course content text from which to generate questions>"
            }

            2. Output:
            - You must generate one or more complete question objects in the following exact JSON format:
            [
                {
                    "question": "<The generated short-answer question>",
                    "difficulty": "<easy | medium | hard>",
                    "mark_scheme": [
                        {
                            "point": "<Specific idea, fact, or reasoning step that earns marks>",
                            "marks": <integer number of marks, usually 1-3>
                        },
                        ...
                    ],
                    "total_marks": <sum of marks in mark_scheme>,
                    "expected_answer": "<A well-written short paragraph (50-100 words) that covers all key points>"
                },
                ...
            ]

            3. Question requirements:
            - Each question must elicit a short, paragraph-style written answer (approximately 50-100 words).
            - Questions should test **understanding, explanation, reasoning, or conceptual comparison**, rather than simple recall.
            - Use phrasing such as "Explain", "Describe", "Compare", "Discuss", or "Why".
            - Avoid yes/no or multiple-choice style phrasing.
            - Each question must be self-contained and understandable without referring back to the original text.
            - Avoid duplicating similar questions.

            4. Difficulty labeling:
            - Assign a difficulty level based on the cognitive effort required:
                - "easy" = Explains or defines a single concept directly from the material.
                - "medium" = Requires connecting two or more ideas or applying a concept to a scenario.
                - "hard" = Requires deeper reasoning, critical evaluation, or synthesis across topics.

            5. Mark scheme requirements:
            - Include **4 to 8** assessable points in total.
            - Each point should describe a distinct concept or reasoning step that a good answer would contain.
            - Each point should be concise (1-2 short sentences).
            - Allocate 1-3 marks per point, depending on importance.
            - The total marks should usually range from **3-10 marks** depending on the difficulty of the question.
            - Ensure points are specific, measurable, and not overly vague.
            - Difficulty awareness:
                - "easy" = Straightforward factual or definitional points.
                - "medium" = Conceptual reasoning or linking multiple ideas.
                - "hard" = Analytical or evaluative reasoning with multiple interdependent points.

            6. Expected answer requirements:
            - Write a concise, coherent paragraph suitable for an academic short-answer assessment.
            - Integrate all points from the mark scheme naturally into the text.
            - Use clear, formal, academic language.
            - Maintain a logical flow from introduction to conclusion.
            - Keep the total length between **50 and 100 words**.
            - Do not list points, instead write continuous prose.
            - Avoid adding extra ideas not in the mark scheme.
            - The answer must cover all points in the mark scheme.

            7. Output formatting:
            - Always output a valid **JSON array of objects** matching the schema above.
            - Do NOT include commentary, reasoning, or text outside the JSON.
            - Ensure valid JSON syntax (no trailing commas, escaped quotes, etc.).
            - Ensure total_marks equals the sum of all marks in the mark_scheme.
            """
    
    def generate_questions(
            self, 
            file_name: str,
            content_dict: Dict[str, str]
        ) -> Dict[str, List[Dict[str, str]]]:
        """
        Generates short-answer questions for a single piece of course content in one LLM call.

        Args:
            file_name (str): The name of the source file the content came from (used for logging and packing into results).
            content_dict (Dict[str, str]): The content dictionary for the file. Expected keys:
                - "text" (str): The extracted course content to generate questions from.
                - "file_path" (str): The path of the source file (packed into each result).
                - "week_dir" (str): The week directory the file belongs to (packed into each result).
        Returns:
            Dict[str, List[Dict[str, str]]]: A dictionary with two keys:
                - "valid_questions" (List[Dict[str, str]]): Successfully generated and validated question dicts.
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

        # Single stage: Generate question, mark scheme, and expected answer all at once
        response_text: str = generate_llm_response(
            client=self.client,
            model_name=self.model_name,
            messages=messages
        )
        
        response_json = extract_json_from_text(response_text)
        if response_json is not None:
            # Validate the response
            try:
                assert response_json is not None and isinstance(response_json, list), "Generated JSON is not a list"
                assert all(isinstance(q, dict) for q in response_json), "Each entry must be a dict"
                
                for q in response_json:
                    assert "question" in q and "difficulty" in q, "Missing 'question' or 'difficulty'"
                    assert "mark_scheme" in q and "total_marks" in q, "Missing 'mark_scheme' or 'total_marks'"
                    assert "expected_answer" in q, "Missing 'expected_answer'"
                    assert q["difficulty"].strip().lower() in ["easy", "medium", "hard"], "Invalid difficulty level"
                    assert isinstance(q["mark_scheme"], list), "mark_scheme must be a list"
                    assert all(isinstance(point, dict) and "point" in point and "marks" in point for point in q["mark_scheme"]), "Invalid mark_scheme structure"
                    assert isinstance(q["total_marks"], int), "total_marks must be an integer"
                    assert q["total_marks"] == sum(point["marks"] for point in q["mark_scheme"]), "total_marks mismatch"
                    assert isinstance(q["expected_answer"], str) and len(q["expected_answer"].strip()) > 0, "expected_answer must be non-empty string"
            except AssertionError as e:
                # raise ValueError(f"Failed to validate generated response: {e}\nResponse: {response_text}")
                return {
                    "valid_questions": [],
                    "failed_questions": []
                }
            
            # print(f"Parsed complete questions JSON:\n{json.dumps(response_json, indent=4)}")
            
            # Pack all data together
            for gen_question_dict in response_json:
                packed_data = {
                    "file_path": content_dict["file_path"],
                    "week": content_dict["week_dir"],
                    "question": gen_question_dict["question"],
                    "difficulty": gen_question_dict["difficulty"].strip().lower(),
                    "bloom_level": gen_question_dict["bloom_level"].strip().lower() if "bloom_level" in gen_question_dict else None,
                    "mark_scheme": gen_question_dict["mark_scheme"],
                    "total_marks": gen_question_dict["total_marks"],
                    "expected_answer": gen_question_dict["expected_answer"],
                }
                valid_questions.append(packed_data)
    
        return {
            "valid_questions": valid_questions,
            "failed_questions": failed_questions
        }

class MultiStageZeroShotSAQGenerator:
    """
    Multi-stage zero-shot SAQ generator.
    - Stage 1: generate question stems with a difficulty label.
    - Stage 2: generate a mark scheme per question, then an expected answer from that mark scheme.
    - Each stage is a separate LLM call; no verification checks are applied.
    """

    def __init__(
            self,
            client:OpenAI,
            model_name:str,
            use_bloom_prompting:bool
        ):
        """
        Initialises the multi-stage zero-shot SAQ generator and its per-stage system instructions.

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
            You are an expert educational content generator specialized in transforming course material into short-answer questions suitable for written assessments.

            Your goal is to produce high-quality, contextually relevant short-answer question prompts from the provided course content. Each question must be labeled 
            with both an appropriate difficulty level and a Bloom's Taxonomy cognitive level.

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
                "extracted_text": "<The extracted course content text from which to generate questions>"
            }

            2. Output:
            You must generate one or more question objects in the following exact JSON format:

            [
                {
                    "question": "<The generated short-answer question>",
                    "difficulty": "<easy | medium | hard>",
                    "bloom_level": "<knowledge | understanding | application | analyze | synthesis | evaluation>"
                }
            ]

            3. Question requirements:
            - Each question must elicit a short, paragraph-style written answer (approximately 50-100 words).
            - Questions should test understanding, explanation, reasoning, application, comparison, or evaluation rather than simple recall (unless explicitly appropriate).
            - Use phrasing such as "Explain", "Describe", "Compare", "Discuss", "Evaluate", or "Why".
            - Avoid yes/no or multiple-choice style phrasing.
            - Each question must be self-contained and understandable without referring back to the original text.
            - Avoid duplicating similar questions.

            4. Bloom level assignment:
            - Assign exactly one Bloom cognitive level per question.
            - The assigned level must reflect the primary cognitive process required to answer the question.
            - The Bloom level must align with the command verb and the depth of reasoning required.
            - If multiple cognitive processes are present, select the dominant one.

            5. Difficulty labeling:
            - Assign difficulty based on cognitive effort:
                - "easy" = Explains or defines a single concept directly from the material (typically remember/understand).
                - "medium" = Requires connecting multiple ideas or applying a concept (often apply/analyze).
                - "hard" = Requires deeper reasoning, evaluation, or synthesis (often evaluate/create).
            - Difficulty and Bloom level should generally align, but Bloom classification must be based strictly on cognitive demand.

            6. Output formatting:
            - Always output a valid JSON array of objects matching the schema above.
            - Do NOT include answers, commentary, or text outside the JSON.
            - Ensure valid JSON syntax (no trailing commas, escaped quotes, etc.).
            """
        else:
            self.QUESTION_GENERATION_SYSTEM_INSTRUCTION = """
            You are an expert educational content generator specialized in transforming course material into short-answer questions suitable for written assessments.

            Your goal is to produce high-quality, contextually relevant short-answer question prompts from the provided course content, each labeled with an appropriate difficulty level.
            
            Follow these rules strictly:

            1. Input:
                You will receive a JSON object of the format:
                    {
                        "extracted_text": "<The extracted course content text from which to generate questions>"
                    }


            2. Output:
            - You must generate one or more question objects in the following exact JSON format:
            [
                {
                    "question": "<The generated short-answer question>",
                    "difficulty": "<easy | medium | hard>"
                },
                ...
            ]

            3. Question requirements:
            - Each question must elicit a short, paragraph-style written answer (approximately 50-100 words).
            - Questions should test **understanding, explanation, reasoning, or conceptual comparison**, rather than simple recall.
            - Use phrasing such as “Explain', “Describe', “Compare', “Discuss', or “Why'.
            - Avoid yes/no or multiple-choice style phrasing.
            - Each question must be self-contained and understandable without referring back to the original text.
            - Avoid duplicating similar questions.

            4. Difficulty labeling:
            - Assign a difficulty level based on the cognitive effort required:
                - "easy" = Explains or defines a single concept directly from the material.
                - "medium" = Requires connecting two or more ideas or applying a concept to a scenario.
                - "hard" =Requires deeper reasoning, critical evaluation, or synthesis across topics.

            5. Output formatting:
            - Always output a valid **JSON array of objects** matching the schema above.
            - Do NOT include answers, commentary, or text outside the JSON.
            - Ensure valid JSON syntax (no trailing commas, escaped quotes, etc.).
            """

        self.MARK_SCHEME_GENERATION_SYSTEM_INSTRUCTION = """
        You are an expert educational content designer specializing in constructing mark schemes for short-answer questions.

        Your goal is to take a given short-answer question (and its difficulty level) and produce a structured mark scheme that identifies key points students should mention and how marks are allocated.

        Follow these rules strictly:

        1. Input:
        - You will receive a JSON object containing:
            {
                "question": "<question text>",
                "difficulty": "<easy | medium | hard>"
            }

        2. Output:
        - You must produce exactly one object in the following JSON format:
            {
                "question": "<same question text>",
                "mark_scheme": [
                    {
                        "point": "<Specific idea, fact, or reasoning step that earns marks>",
                        "marks": <integer number of marks, usually 1-3>
                    },
                    ...
                ],
                "total_marks": <sum of marks in mark_scheme>
            }

        3. Mark scheme requirements:
        - Include **4 to 8** assessable points in total.
        - Each point should describe a distinct concept or reasoning step that a good answer would contain.
        - Each point should be concise (1-2 short sentences).
        - Allocate 1-3 marks per point, depending on importance.
        - The total marks should usually range from **6-10 marks**.
        - Ensure points are specific, measurable, and not overly vague.

        4. Difficulty awareness:
        - "easy" = Straightforward factual or definitional points.
        - "medium" = Conceptual reasoning or linking multiple ideas.
        - "hard" = Analytical or evaluative reasoning with multiple interdependent points.

        5. Output formatting:
        - Always output a **single valid JSON object** (not an array).
        - Do NOT include any commentary, explanations, or text outside the JSON.
        - Ensure valid JSON syntax (no trailing commas, escaped quotes, etc.).
        """

        self.ANSWER_GENERATION_SYSTEM_INSTRUCTION = """
        You are an expert educational content generator specialized in writing model answers for short-answer questions.

        Your goal is to take a question (with its difficulty level) and an accompanying mark scheme, and produce a concise, coherent model answer that fully covers all key points listed in the mark scheme.

        Follow these rules strictly:

        1. Input:
        - You will receive a JSON object containing:
            {
                "question": "<question text>",
                "difficulty": "<easy | medium | hard>",
                "mark_scheme": [
                    {"point": "<point 1>", "marks": <int>},
                    {"point": "<point 2>", "marks": <int>},
                    ...
                ]
            }

        2. Output:
        - You must produce exactly one JSON object in the following format:
            {
                "question": "<same question text>",
                "expected_answer": "<A well-written short paragraph (50-100 words) that covers all key points in the mark scheme clearly and naturally.>"
            }

        3. Expected answer requirements:
        - Write a concise, coherent paragraph suitable for an academic short-answer assessment.
        - Integrate all points from the mark scheme naturally into the text.
        - Use clear, formal, academic language.
        - Maintain a logical flow from introduction to conclusion.
        - Keep the total length between **50 and 100 words**.
        - Do not list points - write continuous prose.
        - Avoid adding extra ideas not in the mark scheme.

        4. Difficulty awareness:
        - "easy" = Focus on clear explanations of core ideas.
        - "medium" = Include connections or examples to show understanding.
        - "hard" = Integrate reasoning, evaluation, or synthesis of multiple concepts.

        5. Output formatting:
        - Always output a **single valid JSON object** (not an array).
        - Do NOT include commentary, reasoning, or explanations outside the JSON.
        - Ensure valid JSON syntax (no trailing commas, escaped quotes, etc.).
        """
    
    def generate_questions(
            self, 
            file_name:str,
            content_dict:Dict[str, str]
        ) -> List[Dict[str, str]]:
        """
        Generates short-answer questions for a single piece of course content across multiple LLM stages.
        - Stage 1 generates question stems with difficulty labels.
        - Stage 2 generates a mark scheme per question and an expected answer from that mark scheme.

        Args:
            file_name (str): The name of the source file the content came from (used for logging and packing into results).
            content_dict (Dict[str, str]): The content dictionary for the file. Expected keys:
                - "text" (str): The extracted course content to generate questions from.
                - "file_path" (str): The path of the source file (packed into each result).
                - "week_dir" (str): The week directory the file belongs to (packed into each result).
        Returns:
            Dict[str, List[Dict[str, str]]]: A dictionary with two keys (the runtime return is this dict, not a bare list):
                - "valid_questions" (List[Dict[str, str]]): Successfully generated questions with mark schemes and expected answers.
                - "failed_questions" (List[Dict[str, str]]): Questions that failed a stage (empty here; failures are skipped).
        """

        valid_questions:List[Dict[str, str]] = []
        failed_questions:List[Dict[str, str]] = [] # TODO: Implement verification checks here?

        extracted_text:str = content_dict["text"]

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

        # Stage 1: Generate short-answer question and assign a difficulty level
        question_response_text:str = generate_llm_response(
            client=self.client,
            model_name=self.model_name,
            messages=messages
            )
        question_json = extract_json_from_text(question_response_text)
        if question_json is None: # Skip if JSON cannot be parsed.
            # print(f"Failed to parse generated questions JSON for file {file_name}. Skipping further processing for this file.\nResponse text was:\n{question_response_text}\n")
            return {
                "valid_questions": valid_questions,
                "failed_questions": failed_questions
            }
        try:
            assert question_json is not None and isinstance(question_json, list), "Generated JSON is not a list"
            assert all(isinstance(q, dict) and "question" in q and "difficulty" in q for q in question_json), "Each question entry must be a dict with 'question' and 'difficulty'"
            assert all(q["difficulty"].strip().lower() in ["easy", "medium", "hard"] for q in question_json), "Each question's difficulty must be one of 'easy', 'medium', or 'hard'"
        except:
            # raise ValueError(f"Failed to parse generated response as JSON: {question_response_text}")
            return {
                "valid_questions": valid_questions,
                "failed_questions": failed_questions
            }
        # print(f"Parsed questions JSON:\n {json.dumps(question_json, indent=4)}")
        
        # Stage 2: Generate a mark scheme for each generated question + generate the expected answer based on the generated mark scheme
        for gen_question_idx, gen_question_dict in enumerate(question_json):

            # print(f"Generated question:", gen_question_dict)
            if "question" not in gen_question_dict or "difficulty" not in gen_question_dict:
                # print(f"Skipping invalid question dict (missing 'question' or 'difficulty'): {gen_question_dict}")
                continue
            question = gen_question_dict["question"].strip()
            difficulty = gen_question_dict["difficulty"].strip().lower()   

            # print(f"Generating mark scheme & answer for question: {question} (Difficulty: {difficulty}) | Question {gen_question_idx+1}/{len(question_json)} in file {file_name}")

            mark_scheme_gen_input_content = {
                "question": question,
                "difficulty": difficulty,
            }
            mark_scheme_gen_input_json = json.dumps(mark_scheme_gen_input_content, indent=4)
            mark_scheme_gen_messages = [
                {"role": "system", "content": self.MARK_SCHEME_GENERATION_SYSTEM_INSTRUCTION},
                {"role": "user", "content": mark_scheme_gen_input_json}
            ]
            mark_scheme_response_text:str = generate_llm_response(
                client=self.client,
                model_name=self.model_name,
                messages=mark_scheme_gen_messages
            )
            mark_scheme_json = extract_json_from_text(mark_scheme_response_text)
            if mark_scheme_json is None: # Skip if JSON cannot be parsed.
                continue

            # print(f"Parsed mark scheme JSON:\n {json.dumps(mark_scheme_json, indent=4)}")
            try:
                assert mark_scheme_json is not None and isinstance(mark_scheme_json, dict), "Mark scheme JSON is not a dict"
                assert "mark_scheme" in mark_scheme_json and "total_marks" in mark_scheme_json, "Mark scheme JSON missing required fields"
                assert isinstance(mark_scheme_json["mark_scheme"], list) and all(isinstance(point, dict) and "point" in point and "marks" in point for point in mark_scheme_json["mark_scheme"]), "Mark scheme must be a list of dicts with 'point' and 'marks'"
                assert isinstance(mark_scheme_json["total_marks"], int) and mark_scheme_json["total_marks"] == sum(point["marks"] for point in mark_scheme_json["mark_scheme"]), "Total marks must equal sum of marks in mark scheme"
            except Exception as e:
                # raise ValueError(f"Generated mark scheme JSON failed validation: {e}\nMark Scheme JSON: {json.dumps(mark_scheme_json, indent=4)}")
                continue # Skip to next question if mark scheme JSON is invalid
            
            # Generate the expected answer based on the generated mark scheme
            answer_gen_input_content = {
                "question": question,
                "difficulty": difficulty,
                "mark_scheme": mark_scheme_json["mark_scheme"]
            }
            answer_gen_input_json = json.dumps(answer_gen_input_content, indent=4)
            answer_gen_messages = [
                {"role": "system", "content": self.ANSWER_GENERATION_SYSTEM_INSTRUCTION},
                {"role": "user", "content": answer_gen_input_json}
            ]
            answer_response_text:str = generate_llm_response(
                client=self.client,
                model_name=self.model_name,
                messages=answer_gen_messages
            )
            answer_json = extract_json_from_text(answer_response_text)
            if answer_json is None: # Skip if JSON cannot be parsed.
                continue
            # print(f"Parsed answer JSON:\n {json.dumps(answer_json, indent=4)}")
        
            try:
                assert answer_json is not None and isinstance(answer_json, dict), "Answer JSON is not a dict"
                assert "expected_answer" in answer_json, "Answer JSON missing required fields"
                assert isinstance(answer_json["expected_answer"], str) and len(answer_json["expected_answer"].strip()) > 0, "Expected answer must be a non-empty string"
            except Exception as e:
                # raise ValueError(f"Generated answer JSON failed validation: {e}\nAnswer JSON: {json.dumps(answer_json, indent=4)}")
                continue
            
            # Pack all data together
            packed_data = {
                "file_path": content_dict["file_path"],
                "week": content_dict["week_dir"],
                "question": question,
                "difficulty": difficulty,
                "bloom_level": gen_question_dict["bloom_level"].strip().lower() if "bloom_level" in gen_question_dict else None,
                "mark_scheme": mark_scheme_json["mark_scheme"], # List of dicts with "point" and "marks"
                "total_marks": mark_scheme_json["total_marks"],
                "expected_answer": answer_json["expected_answer"],
            }
            valid_questions.append(packed_data)
    
        return {
            "valid_questions": valid_questions,
            "failed_questions": failed_questions
        }

class BloomSAQGenerator:
    """
    Proposed Bloom-aware SAQ generation pipeline.
    - Generates question stems for each Bloom's taxonomy level using `GeneratorSystem` and the per-level system instructions.
    - Runs verification at each stage via `VerifierSystem`: phase-one checks (Bloom alignment + course relevance), then mark-scheme and expected-answer coverage checks.
    - Only questions that pass every check are returned as valid.
    """

    def __init__(
            self,
            client:OpenAI,
            model_name:str
        ):
        """
        Initialises the Bloom-aware SAQ pipeline along with its generator and verifier systems.

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
        Generates and verifies short-answer questions for a single piece of course content across all Bloom's levels.
        - For each Bloom level: generates question stems, runs phase-one verification, then generates and verifies the mark scheme and expected answer.

        Args:
            file_name (str): The name of the source file the content came from (used for logging and packing into results).
            content_dict (Dict[str, str]): The content dictionary for the file. Expected keys:
                - "text" (str): The extracted course content to generate questions from.
                - "file_path" (str): The path of the source file (packed into each result).
                - "week_dir" (str): The week directory the file belongs to (packed into each result).
        Returns:
            Dict[str, List[Dict[str, str]]]: A dictionary with two keys:
                - "valid_questions" (List[Dict[str, str]]): Questions that passed every generation and verification stage.
                - "failed_questions" (List[Dict[str, str]]): Information dicts for questions that failed a verification check.
        """

        valid_questions:List[Dict[str, str]] = []
        failed_questions:List[Dict[str, str]] = []

        extracted_text:str = content_dict["text"]

        # Iterate over each Bloom's taxonomy level and generate questions
        for j, (bloom_level, question_gen_system_instruction) in enumerate(QUESTION_GEN_MAPPINGS.items()):
            # print(f"Generating questions for Bloom's level: {bloom_level} {j+1}/{len(QUESTION_GEN_MAPPINGS)}")
            # print(f"Extracted text:\n {extracted_text}")

            # Stage 1: Generate short-answer question and assign a difficulty level
            question_json = self.generator_system.generate_question_stems(
                extracted_text=extracted_text,
                bloom_level=bloom_level,
                question_gen_system_instruction=question_gen_system_instruction
            )
            if question_json is None:
                # print(f"Failed to generate questions for Bloom's level: {bloom_level}. Skipping further processing for this level.\n")
                continue
            # print(f"Parsed questions JSON:\n {json.dumps(question_json, indent=4)}")
            # print()
            
            # Phase one verification checks (Bloom level alignment and course content relevance)
            phase_one_result:Tuple[List[Dict[str, str]], Set[int]] = self.verifier_system.perform_phase_one_check(
                question_json=question_json,
                extracted_text=extracted_text,
                file_name=file_name,
            )
            additional_failed_questions, failed_indices = phase_one_result
            failed_questions.extend(additional_failed_questions)
            if len(additional_failed_questions) > 0:
                # Filter out the questions that failed phase one checks
                filtered_question_json:List[Dict[str, str]] = [q for idx, q in enumerate(question_json) if idx not in failed_indices]
                # print(f"Previous length: {len(question_json)} | Filtered length: {len(filtered_question_json)} after removing failed questions.\n")
            else:
                filtered_question_json = question_json # No questions failed phase one checks
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
                    extracted_text=extracted_text
                )
                if mark_scheme_json is None:
                    # print(f"Failed to generate mark scheme for question: {question}. Skipping further processing for this question.\n")
                    continue

                # Mark scheme coverage verification check
                additional_failed_questions:List[Dict[str, str]] = self.verifier_system.perform_mark_scheme_coverage_check(
                    question_dict=gen_question_dict,
                    mark_scheme_dict=mark_scheme_json,
                    bloom_level=bloom_level,
                    difficulty=difficulty,
                    extracted_text=extracted_text,
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
                    # print(f"Failed to generate expected answer for question: {question}. Skipping further processing for this question.\n")
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
                    "question": question,
                    "difficulty": difficulty,
                    "bloom_level": bloom_level,
                    "mark_scheme": mark_scheme_json["mark_scheme"], # List of dicts with "point" and "marks"
                    "total_marks": mark_scheme_json["total_marks"],
                    "expected_answer": expected_answer_json["expected_answer"],
                }
                valid_questions.append(packed_data)
        
        return {
            "valid_questions": valid_questions,
            "failed_questions": failed_questions
        }