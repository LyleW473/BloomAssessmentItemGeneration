import time
import random
import json
import re

from openai import OpenAI, APIStatusError
from google.genai import errors
from typing import List, Dict

def extract_json_from_text(response_text:str) -> Dict[str,str]:
    """
    Extracts JSON object from a given text response. Assumes the JSON is enclosed within triple backticks (```).

    Args:
        response_text (str): The text response containing the JSON object.
    Returns:
        Dict[str, str]: The extracted JSON object as a dictionary. Returns None if extraction or parsing fails.
    """
    try:
        response_text = re.sub(r"^```json\s*|\s*```$", "", response_text.strip())
        answer_json:Dict[str, str] = json.loads(response_text)
    except Exception as e:
        return None
    return answer_json

def generate_llm_response(
                    client:OpenAI,
                    model_name:str,
                    messages:List[Dict[str, str]],
                    max_attempts:int=50,
                    base_delay:float=1.0,
                    ) -> str:
    """
    Generates a response from the LLM API with retries and exponential backoff for handling server unavailability errors.

    Args:
        client (OpenAI): The OpenAI client instance that supports LLM responses for multiple models including Gemini.
        model_name (str): The name of the model to use for generating the response (must be supported by the endpoint).
        messages (List[Dict[str, str]]): The list of messages to send to the model for generating a response. e.g. [{"role": "user", "content": "Hello"}]
        max_attempts (int): Maximum number of attempts to get a response.
        base_delay (float): Base delay in seconds for exponential backoff.
    """

    retryable_codes = [408, 429, 500, 502, 503, 504] # HTTP status codes that are considered retryable
    attempt_num = 1        
    completed = False
    while not completed and attempt_num <= max_attempts:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            completed = True

        except (errors.ServerError, APIStatusError) as e: # Errors: [ServerError = Server unavailable (e.g., 503 errors)], [APIStatusError = HTTP errors from OpenAI API e.g., 408]
            status_code = getattr(e, 'status_code', None)

            print(status_code, type(status_code))

            # Exponential backoff with jitter (for server unavailability errors)
            if status_code in retryable_codes:
                wait_time = min(base_delay * (2 ** attempt_num) + random.uniform(0, 0.5), 600) # Cap wait time to 10 minutes
                print(f"Received {status_code} error from server. Waiting for {wait_time:.2f} seconds before retrying... Attempt number: {attempt_num}")
                time.sleep(wait_time)
            else:
                raise
        attempt_num += 1

    if not completed and attempt_num > max_attempts:
        print(f"Failed to get response from LLM API after {max_attempts} attempts.")
        print("Last error:", e)
        raise RuntimeError(f"Failed to get response from LLM API after {max_attempts} attempts.")
    
    response_text = response.choices[0].message.content
    return response_text

def generate_llm_json_response(client:OpenAI, model_name:str, system_instruction:str, prompt_text:str) -> Dict[str, str]:
    """
    Generates a response from the LLM API and extracts a JSON object from the response text.
    - Essentially a helper function that combines the generation of LLM response and extraction of JSON from the response text.

    Args:
        client (OpenAI): The OpenAI client instance that supports LLM responses for multiple models including Gemini.
        model_name (str): The name of the model to use for generating the response (must be supported by the endpoint).
        system_instruction (str): The system instruction to provide context to the model for generating the response.
        prompt_text (str): The user prompt text to send to the model for generating the response.
    Returns:
        Dict[str, str]: The extracted JSON object from the LLM response as a dictionary. Returns None if extraction or parsing fails.
    """
    response_text:str = generate_llm_response(
        client=client,
        model_name=model_name,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_text},
        ]
    )
    extracted_json = extract_json_from_text(response_text=response_text)
    return extracted_json