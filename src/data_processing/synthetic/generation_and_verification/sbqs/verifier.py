"""
SBQ (scenario-based question) verifier system.
- Defines `VerifierSystem`, which currently reuses the SAQ verification checks for scenario-based questions.
- Inherits all phase-one and coverage checks from the SAQ `VerifierSystem`.
"""
import json
from openai import OpenAI
from typing import List, Dict, Any
from src.llm_response_generation.functions import (generate_llm_json_response)
from src.data_processing.synthetic.generation_and_verification.saqs.verifier import VerifierSystem as BaseVerifierSystemSAQ # To inherit verifying methods for mark scheme and answer

class VerifierSystem(BaseVerifierSystemSAQ):
    """
    A class to perform verification checks on generated synthetic assessment questions and related content using LLMs.

    This class encapsulates methods to verify various aspects of the generated content, including:
    - Bloom's taxonomy level alignment for questions.
    - Relevance of questions to the course content.

    + (unique for SBQs):
    - None currently: SBQ verification reuses the checks inherited from the SAQ VerifierSystem.
    """

    def __init__(self, client:OpenAI, model_name:str):
        """
        Initalises the VerifierSystem with the given OpenAI client and model name.

        Args:
            client (OpenAI): The OpenAI client instance for making API calls.
            model_name (str): The name of the model to use for verification tasks.
        """
        super().__init__(client, model_name)
