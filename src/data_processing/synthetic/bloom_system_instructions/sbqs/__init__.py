"""
Re-exports the SBQ (scenario-based question) Bloom system instructions.
- Exposes the scenario generation, critique, and refinement instructions from `generation`.
- Exposes `QUESTION_GEN_MAPPINGS` (per-Bloom-level question-generation instructions).
"""
from .generation import (
    SCENARIO_GENERATION_INSTRUCTION,
    SCENARIO_CRITIQUE_INSTRUCTION,
    SCENARIO_REFINEMENT_INSTRUCTION,
    QUESTION_GEN_MAPPINGS
)
# from .verification import ()