"""
Re-exports the shared Bloom system instructions used across question types.
- Exposes `BLOOM_LEVEL_TO_DIFFICULTY_MAPPING` and the mark-scheme / answer generation instructions from `generation`.
- Exposes the course-classification, Bloom-level, and course-relevance verification instructions from `verification`.
- Exposes the content-compression preprocessing instruction from `preprocessing`.
"""
from .generation import (
    BLOOM_LEVEL_TO_DIFFICULTY_MAPPING,
    MARK_SCHEME_GENERATION_SYSTEM_INSTRUCTION,
    ANSWER_GENERATION_SYSTEM_INSTRUCTION
)

from .verification import (
    COURSE_CONTENT_CLASSIFICATION_SYSTEM_INSTRUCTION,
    BLOOM_LEVEL_VERIFICATION_SYSTEM_INSTRUCTION,
    COURSE_RELEVANCE_VERIFICATION_SYSTEM_INSTRUCTION,
)

from .preprocessing import (
    COMPRESSION_INSTRUCTION,
)