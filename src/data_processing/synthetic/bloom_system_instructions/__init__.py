"""
Aggregates the Bloom's-taxonomy system instructions (prompt templates) for synthetic assessment generation.
- Re-exports the SAQ, SBQ, and shared generation/verification instruction constants and mappings.
- Importing this package makes every instruction constant available from a single namespace.
"""
from .saqs import * # Import all generation and verification instructions for SAQs
from .sbqs import * # Import all generation and verification instructions for SBQs
from .shared import * # Import shared verification instructions