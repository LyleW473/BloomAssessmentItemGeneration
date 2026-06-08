"""
Global path and directory constants used across the project.
- Defines the base data directories and file paths for raw, processed, and synthetic assessment data.
- Centralises directory names so scripts and pipelines reference a single source of truth.

All data paths are anchored to the project root (the directory containing `src/`),
computed from this file's own location. This makes them absolute, so scripts work
regardless of the current working directory they are launched from (e.g. `scripts/`).
"""
import os

# Project root = parent of the `src/` package that contains this file.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

BATCH_PRINT_INTERVAL=1

GLOBAL_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")

GLOBAL_DATA_DIR = os.path.join(PROJECT_ROOT, "all_data")
MATHDIAL_DATA_DIR = f"{GLOBAL_DATA_DIR}/mathdial"

LAW_DATA_DIR = f"{GLOBAL_DATA_DIR}/law"
LAW_DATA_HUMAN_PATH = f"{LAW_DATA_DIR}/law_qa_human.json"
LAW_DATA_LLM_PATH = f"{LAW_DATA_DIR}/law_qa_llm.json"

AI_DATA_DIR = f"{GLOBAL_DATA_DIR}/ai"
AI_DATA_PATH = f"{AI_DATA_DIR}/ai_qa.json"


# Synthetic data generation
SYNTHETIC_DATA_DIR = f"{GLOBAL_DATA_DIR}/synthetic"
SYNTHETIC_DATA_BASE_COMPONENTS_DIR = f"{SYNTHETIC_DATA_DIR}/base_components"
SYNTHETIC_DATA_SCENARIOS_PATH = f"{SYNTHETIC_DATA_BASE_COMPONENTS_DIR}/scenarios.json"
SYNTHETIC_DATA_GENERATED_SCENARIOS_DIR = f"{SYNTHETIC_DATA_BASE_COMPONENTS_DIR}/generated_scenarios"
SYNTHETIC_DATA_GENERATED_CONVERSATIONS_DIR = f"{SYNTHETIC_DATA_DIR}/generated_conversations"
SYNTHETIC_ASSESSMENT_DATA_DIR = f"{SYNTHETIC_DATA_DIR}/generated_assessment_student_responses"
SYNTHETIC_ASSESSMENT_DATA_QUESTIONS_DIR = f"{SYNTHETIC_DATA_DIR}/generated_assessment_student_questions"

BASE_ASSESSMENT_QUESTIONS_DATA_DIR = f"{GLOBAL_DATA_DIR}/assessment" # Where all the course content is stored so that we can generate questions from it
BASE_EXTRACTED_QUESTIONS_DATA_DIR = f"{GLOBAL_DATA_DIR}/processed_assessment" # Contains the extracted texts from course content, and the generated questions
