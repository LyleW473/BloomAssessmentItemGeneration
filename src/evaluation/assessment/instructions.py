TRUE_BLOOM_LEVEL_INSTRUCTION = """
You are an expert educational assessor.

You will be given a JSON input with:
{
  "question": "<string>",
  "extracted_text": "<string>"
}

Bloom levels:
- knowledge:
  Recall or state a fact, definition, formula, or concept without transformation.

- understanding:
  Explain, summarize, or interpret a concept in one's own words.
  Requires demonstrating meaning, not just recall.

- application:
  Use a concept, method, or rule in a specific context or example.
  Requires applying knowledge to a scenario or problem.

- analyze:
  Break down a concept, compare components, identify relationships, causes, assumptions, or structure.

- evaluation:
  Make a justified judgment using explicit criteria.
  Requires defending or critiquing with reasoning.

- synthesis:
  Create, design, or propose something new by combining elements meaningfully.

Task:
Infer the single TRUE Bloom level required by the question stem (based on what the learner must DO).

Rules:
- Judge the minimum cognitive process REQUIRED for full marks, not the surface verb alone.
- If the question is "create", output "synthesis".
- If the question is "evaluate", output "evaluation".
- A question is application only if it requires using knowledge in a concrete scenario, case, or problem context.
- A question is evaluation only if it requires making and defending a judgment using criteria.
- A question is analyze only if it requires distinguishing parts, relationships, trade-offs, causes, or structure.
- If a question merely asks for explanation of a concept, prefer understanding.
- Only output one of: knowledge, understanding, application, analyze, evaluation, synthesis
- Output ONLY one label.

Label:
"""

# Criterion 1: Course Grounding (STEM-only)
COURSE_GROUNDING = """
You are an expert educational assessor.

Input JSON:
{
  "question": "<string>",
  "true_bloom_level": "<knowledge | understanding | application | analyze | evaluation | synthesis>",
  "declared_bloom_level": "<knowledge | understanding | application | analyze | evaluate | create | unknown>",
  "extracted_text": "<string>"
}

Scoring principle:
- Start from 2.
- Downgrade to 1 if any meaningful weakness is present.
- Downgrade to 0 if there is a major flaw.
- Score 2 should be reserved for strong, exam-ready cases with no meaningful weaknesses.
- If the item is merely acceptable or broadly correct, prefer 1 rather than 2.

Score COURSE GROUNDING: how strongly does the question depend on extracted_text?
- 2 = strongly grounded; answering well requires specific concepts, distinctions, terminology, examples, or relationships that are present in extracted_text
- 1 = topically related but mostly generic; a student could answer well from general domain knowledge without much dependence on extracted_text
- 0 = weakly grounded or off-scope; not meaningfully traceable to extracted_text

Downgrade guidance:
- If the question could be answered well from a standard textbook or general background knowledge alone, prefer 1.
- Only assign 2 when extracted_text materially constrains or supports the expected answer.

Output rules:
- Output the score ONLY: 0, 1, or 2.
- Do NOT invent facts outside extracted_text when judging grounding.
- Be strict; if uncertain, prefer lower.

Score:
"""

# Criterion 2: Clarity and Specificity (Stem-only)
CLARITY = """
You are an expert educational assessor.

Input JSON:
{
  "question": "<string>",
  "true_bloom_level": "<knowledge | understanding | application | analyze | evaluation | synthesis>",
  "declared_bloom_level": "<knowledge | understanding | application | analyze | evaluate | create | unknown>",
  "extracted_text": "<string>"
}

Scoring principle:
- Start from 2.
- Downgrade to 1 if any meaningful weakness is present.
- Downgrade to 0 if there is a major flaw.
- Score 2 should be reserved for strong, exam-ready cases with no meaningful weaknesses.
- If the item is merely acceptable or broadly correct, prefer 1 rather than 2.

Score CLARITY: is the question precise and tightly scoped enough that a competent student would know exactly what is required?
- 2 = precise, self-contained, tightly scoped, and unambiguous
- 1 = understandable but somewhat broad, underspecified, or open to minor scope variation
- 0 = unclear, missing essential information, or open to multiple substantially different interpretations

Downgrade guidance:
- Broad prompts using verbs such as "discuss" or "explain" can still score 2 only if the scope is clearly bounded.
- If multiple reasonable answer scopes exist, prefer 1.

Output rules:
- Output the score ONLY: 0, 1, or 2.
- Be strict; if uncertain, prefer lower.

Score:
"""

# Criterion 3: Single Task Integrity (STEM-only)
SINGLE_TASK_INTEGRITY = """
You are an expert educational assessor.

Input JSON:
{
  "question": "<string>",
  "true_bloom_level": "<knowledge | understanding | application | analyze | evaluation | synthesis>",
  "declared_bloom_level": "<knowledge | understanding | application | analyze | evaluate | create | unknown>",
  "extracted_text": "<string>"
}

Scoring principle:
- Start from 2.
- Downgrade to 1 if any meaningful weakness is present.
- Downgrade to 0 if there is a major flaw.
- Score 2 should be reserved for strong, exam-ready cases with no meaningful weaknesses.
- If the item is merely acceptable or broadly correct, prefer 1 rather than 2.

Score SINGLE TASK INTEGRITY: does the question ask for one coherent task rather than multiple separable tasks?
- 2 = one coherent task; any sub-parts are tightly integrated and necessary to the same response
- 1 = mostly one task but includes separable sub-parts or multiple requested actions
- 0 = clearly multiple independent tasks that could be answered separately

Downgrade guidance:
- If the stem asks the student to do more than one thing (e.g., define + compare, explain + justify, describe + provide example), prefer 1 unless these are inseparable parts of one coherent act.

Output rules:
- Output the score ONLY: 0, 1, or 2.
- Be strict; if uncertain, prefer lower.

Score:
"""

# Criterion 4: Objective Gradability (STEM-only)
OBJECTIVE_GRADABILITY_STEM = """
You are an expert educational assessor.

Input JSON:
{
  "question": "<string>",
  "true_bloom_level": "<knowledge | understanding | application | analyze | evaluation | synthesis>",
  "declared_bloom_level": "<knowledge | understanding | application | analyze | evaluate | create | unknown>",
  "extracted_text": "<string>"
}

Scoring principle:
- Start from 2.
- Downgrade to 1 if any meaningful weakness is present.
- Downgrade to 0 if there is a major flaw.
- Score 2 should be reserved for strong, exam-ready cases with no meaningful weaknesses.
- If the item is merely acceptable or broadly correct, prefer 1 rather than 2.

Score OBJECTIVE GRADABILITY (STEM-ONLY):
Could answers be graded consistently from the stem alone, without relying on a hidden rubric?
- 2 = the stem clearly implies what a full-credit answer must contain; answer boundaries are constrained
- 1 = broadly gradable, but there are multiple reasonable answer structures or unclear boundaries for full credit
- 0 = too open-ended, subjective, or underspecified for consistent grading from the stem alone

Downgrade guidance:
- If different markers could reasonably expect different answer components, prefer 1.
- If the stem uses open-ended verbs without clear limits or criteria, prefer 1 or 0.
- Do not assume the mark scheme when judging this criterion.

Output rules:
- Output the score ONLY: 0, 1, or 2.
- Do NOT consider any mark scheme (stem-only).
- Be strict; if uncertain, prefer lower.

Score:
"""

# Criterion 5: Objective Gradability (TRIPLE-AWARE)
OBJECTIVE_GRADABILITY_TRIPLE = """
You are an expert educational assessor.

Input JSON:
{
  "question": "<string>",
  "true_bloom_level": "<knowledge | understanding | application | analyze | evaluation | synthesis>",
  "declared_bloom_level": "<knowledge | understanding | application | analyze | evaluate | create | unknown>",
  "extracted_text": "<string>",
  "mark_scheme": [
    {"point": "<string>", "marks": <int>},
    ...
  ],
  "expected_answer": "<string>"
}

Scoring principle:
- Start from 2.
- Downgrade to 1 if any meaningful weakness is present.
- Downgrade to 0 if there is a major flaw.
- Score 2 should be reserved for strong, exam-ready cases with no meaningful weaknesses.
- If the item is merely acceptable or broadly correct, prefer 1 rather than 2.

Score OBJECTIVE GRADABILITY (TRIPLE-AWARE):
Can this question be graded consistently using this mark scheme and expected answer?
- 2 = answer requirements are explicit, points are distinct, and different markers would likely award similar marks
- 1 = generally gradable, but some points overlap, are loosely phrased, or leave room for inconsistent judgment
- 0 = unreliable for consistent marking; major overlap, vagueness, hidden assumptions, or subjective judgment remains

Downgrade guidance:
- Prefer 1 if multiple mark scheme points reward nearly the same idea.
- Prefer 1 if a point is conceptually right but not measurably assessable.
- Prefer 0 if success depends on unstated assumptions or discretionary interpretation.

Output rules:
- Output the score ONLY: 0, 1, or 2.
- Be strict; if uncertain, prefer lower.

Score:
"""

MARK_RANGES = { # Expected ranges of marks for each true Bloom level.
    "knowledge": (1, 3),
    "understanding": (3, 5),
    "application": (4, 8),
    "analyze": (6, 10),
    "evaluation": (6, 10),
    "synthesis": (8, 12)
}

# Criterion 6: Mark Scheme Quality (TRIPLE-AWARE)
MARK_SCHEME_QUALITY = """
You are an expert educational assessor.

Input JSON:
{
  "question": "<string>",
  "true_bloom_level": "<knowledge | understanding | application | analyze | evaluation | synthesis>",
  "declared_bloom_level": "<knowledge | understanding | application | analyze | evaluate | create | unknown>",
  "extracted_text": "<string>",
  "mark_scheme": [
    {"point": "<string>", "marks": <int>},
    ...
  ],
  "expected_answer": "<string>"
}

Scoring principle:
- Start from 2.
- Downgrade to 1 if any meaningful weakness is present.
- Downgrade to 0 if there is a major flaw.
- Score 2 should be reserved for strong, exam-ready cases with no meaningful weaknesses.
- If the item is merely acceptable or broadly correct, prefer 1 rather than 2.

IMPORTANT:
- Always evaluate structural and cognitive alignment using the TRUE Bloom level, not the declared Bloom level.
- Do NOT re-check whether the total marks fall within the allowed range for the TRUE Bloom level; that hard constraint has already been checked separately.
- Focus only on the remaining quality questions below.

----------------------------------------------------------------------
Step 1: Check Bloom-level structural and cognitive fit using TRUE Bloom level

Expected qualitative templates:

KNOWLEDGE:
- factual recall only
- no explanation beyond basic statement/definition
- points should be short, direct, and strictly factual

UNDERSTANDING:
- explanation, description, or interpretation
- no deep reasoning, critique, or multi-step analysis
- points should show comprehension rather than mere recall

APPLICATION:
- concrete use of knowledge in a scenario, procedure, or method
- points should reflect doing or applying, not just explaining

ANALYZE:
- comparison, breakdown, relationships, causes, assumptions, or structure
- points should reflect decomposition or contrast

EVALUATION:
- judgment, critique, or preference justified by criteria
- points should reflect evaluative reasoning, not just explanation

SYNTHESIS:
- proposal, design, or creation of something new with justification
- points should reflect constructive generation, not just comparison

Downgrade guidance:
- If the point style is noticeably shallower or deeper than the TRUE Bloom level, downgrade.
- If the scheme mostly fits but shows slight depth drift, prefer 1.
- If the scheme is clearly built for the wrong cognitive level, downgrade to 0.

----------------------------------------------------------------------
Step 2: Check conceptual coverage and relevance

- Do the points collectively cover the key ideas needed for a strong answer?
- Are all points directly relevant to the question?

Downgrade guidance:
- Missing an essential idea -> downgrade to 1
- Major conceptual gaps -> downgrade to 0
- Generic filler not clearly required by the question -> downgrade

----------------------------------------------------------------------
Step 3: Check precision and assessability

- Are points specific, measurable, and markable?
- Could two markers consistently award marks using these points?

Downgrade guidance:
- Vague phrases such as "good understanding" or "appropriate use" -> downgrade
- Points that are not independently assessable -> downgrade
- Overly broad or ambiguous phrasing -> downgrade

----------------------------------------------------------------------
Step 4: Check redundancy and independence of points

- Are points distinct and non-overlapping?

Downgrade guidance:
- If multiple points express the same idea in different wording -> downgrade to 1
- If redundancy significantly weakens the scheme or inflates credit -> downgrade to 0

----------------------------------------------------------------------
Step 5: Check alignment with the question

- Does every point directly answer the question asked?

Downgrade guidance:
- If points drift beyond the scope of the question -> downgrade
- If the scheme answers a slightly different question -> downgrade to 0

----------------------------------------------------------------------
Step 6: Check grounding to extracted_text

- Are points supported by extracted_text or standard course knowledge?

Downgrade guidance:
- If key points are not supported or are invented -> downgrade
- If largely generic and not tied to course material -> downgrade to 1

----------------------------------------------------------------------
Final scoring:

- 2 = well aligned to the TRUE Bloom level, conceptually well covered, precise, non-redundant, and exam-ready
- 1 = broadly usable but with minor issues in depth, coverage, precision, redundancy, or alignment
- 0 = major mismatch in cognitive level, major conceptual gaps, vagueness, redundancy, or clear misalignment

Output rules:
- Output the score ONLY: 0, 1, or 2.
- Be strict; do NOT reward verbosity.

Score:
"""

# Criterion 7: Answer Fidelity (TRIPLE-AWARE)
ANSWER_FIDELITY = """
You are an expert educational assessor.

Input JSON:
{
  "question": "<string>",
  "true_bloom_level": "<knowledge | understanding | application | analyze | evaluation | synthesis>",
  "declared_bloom_level": "<knowledge | understanding | application | analyze | evaluate | create | unknown>",
  "extracted_text": "<string>",
  "mark_scheme": [
    {"point": "<string>", "marks": <int>},
    ...
  ],
  "expected_answer": "<string>"
}

Scoring principle:
- Start from 2.
- Downgrade to 1 if any meaningful weakness is present.
- Downgrade to 0 if there is a major flaw.
- Score 2 should be reserved for strong, exam-ready cases with no meaningful weaknesses.
- If the item is merely acceptable or broadly correct, prefer 1 rather than 2.

Score ANSWER FIDELITY:
- 2 = covers all substantive mark scheme points faithfully and introduces no unsupported claims
- 1 = mostly faithful, but compresses, omits, merges, or adds some content in a way that weakens exact alignment
- 0 = significant omissions, contradictions, or substantial unsupported content

Downgrade guidance:
- Prefer 1 if two or more scheme points are merged so heavily that separate coverage becomes unclear.
- Prefer 1 if a required point is only implied rather than clearly realized.
- Judge fidelity strictly relative to the mark scheme, not to general correctness.

Output rules:
- Output the score ONLY: 0, 1, or 2.
- Judge fidelity relative to mark_scheme (not relative to general knowledge).
- Be strict; if uncertain, prefer lower.

Score:
"""

# SBQ-specific criteria:
SCENARIO_RELEVANCE_NECESSITY = """
You are an expert educational assessor.

Input JSON:
{
  "scenario": "<string>",
  "question": "<string>",
  "true_bloom_level": "<knowledge | understanding | application | analyze | evaluation | synthesis>",
  "declared_bloom_level": "<knowledge | understanding | application | analyze | evaluate | create | unknown>",
  "extracted_text": "<string>"
}

Scoring principle:
- Start from 2.
- Downgrade to 1 if any meaningful weakness is present.
- Downgrade to 0 if there is a major flaw.
- Score 2 should be rare and reserved for cases where the scenario is genuinely indispensable.
- If the item is merely acceptable or broadly correct, prefer 1 rather than 2.

Core test (CRITICAL):
- Ask: if the scenario is replaced with a generic version (e.g., "a model is trained on a dataset"),
  would the required answer materially change in content, reasoning, or conclusions?
  - If NO → do NOT assign 2.

Score SCENARIO RELEVANCE & NECESSITY:
Does the question genuinely depend on the scenario, rather than merely being decorated by it?

- 2 = indispensable; the correct answer depends on specific details from the scenario (e.g., values, constraints, structure, outcomes, trade-offs), and removing or genericizing the scenario would materially change what must be explained, calculated, or justified
- 1 = relevant but not essential; the scenario provides helpful context or realism, but the question can be answered largely from general course knowledge without using its specific details
- 0 = decorative or redundant; the scenario can be removed entirely without affecting what constitutes a correct answer

Downgrade guidance:
- If the answer can be produced without referencing any concrete detail from the scenario → prefer 1 or 0
- If the scenario can be replaced with a generic placeholder without changing the reasoning path → score 1, not 2
- If the scenario adds only narrative flavour (e.g., animals, healthcare, e-commerce) without constraining the answer → score 1
- If the scenario does not affect what earns marks in the mark scheme → score 1 or 0
- If the scenario is entirely unused → score 0

Additional strictness:
- For TRUE Bloom levels of application, analyze, evaluation, or synthesis:
  - if the scenario is not required for reasoning → this is a major flaw → score 0
- If the scenario includes hints, conclusions, or embedded reasoning that reduce cognitive demand → downgrade

Output rules:
- Output the score ONLY: 0, 1, or 2.
- Be strict; if uncertain, prefer lower.

Score:
"""

SCENARIO_GROUNDING = """
You are an expert educational assessor.

Input JSON:
{
  "scenario": "<string>",
  "question": "<string>",
  "true_bloom_level": "<knowledge | understanding | application | analyze | evaluation | synthesis>",
  "declared_bloom_level": "<knowledge | understanding | application | analyze | evaluate | create | unknown>",
  "extracted_text": "<string>"
}

Scoring principle:
- Start from 2.
- Downgrade to 1 if any meaningful weakness is present.
- Downgrade to 0 if there is a major flaw.
- Score 2 should be rare and reserved for scenarios clearly traceable to extracted_text.
- If the scenario is merely plausible or generic, prefer 1.

Core test (CRITICAL):
- Ask: could this scenario appear unchanged in a different course or textbook without relying on this extracted_text?
  - If YES → do NOT assign 2.
- Ask: are the scenario's specific elements (not just topic) supported by extracted_text?
  - If NO → do NOT assign 2.

Score SCENARIO GROUNDING:
How well is the scenario grounded in the course content (extracted_text), rather than merely being plausible in the general domain?

- 2 = specifically grounded; the scenario contains concrete elements (e.g., metrics, relationships, data conditions, procedures, constraints, or behaviours) that are directly supported by extracted_text, and would be noticeably weaker or different without that source material
- 1 = broadly plausible and domain-relevant, but generic; it fits the topic but is not tied to specific details from extracted_text and could be reused across many contexts
- 0 = weakly grounded, unsupported, inconsistent, or off-topic relative to extracted_text

Downgrade guidance:
- If the scenario is just a standard ML example (e.g., "predict disease", "classify images") → score 1, not 2
- If the scenario relies only on high-level topic overlap (e.g., “accuracy”, “model”, “dataset”) → score 1
- If the scenario could be reused across different courses with minimal change → prefer 1
- If there is no clear trace from extracted_text to the specific scenario details → score 1
- If the scenario introduces unsupported concepts, assumptions, or facts → downgrade
- If the scenario includes explanation, judgement, or answer cues → downgrade
- If inconsistent with extracted_text → score 0

Additional strictness:
- Plausibility ≠ grounding; realistic scenarios should still receive 1 unless specifically supported
- Score 2 only when extracted_text materially determines the structure or content of the scenario

Output rules:
- Output the score ONLY: 0, 1, or 2.
- Do NOT invent facts outside extracted_text when judging grounding.
- Be strict; if uncertain, prefer lower.

Score:
"""

CRITERION_INSTRUCTIONS_MAPPING = {
    "stem": {
        "course_grounding": COURSE_GROUNDING,
        "clarity": CLARITY,
        "single_task_integrity": SINGLE_TASK_INTEGRITY,
        "objective_gradability_stem": OBJECTIVE_GRADABILITY_STEM,
    },
    "triple": {
        "objective_gradability_triple": OBJECTIVE_GRADABILITY_TRIPLE,
        "mark_scheme_quality": MARK_SCHEME_QUALITY,
        "answer_fidelity": ANSWER_FIDELITY,
    },
    "scenario": {
        "scenario_relevance_necessity": SCENARIO_RELEVANCE_NECESSITY,
        "scenario_grounding": SCENARIO_GROUNDING,
    }
}

# MARK_SCHEME_QUALITY_SCORING_SYSTEM_INSTRUCTION = """
# You are an expert educational assessor specializing in evaluating the quality of generated mark schemes.

# Your task is to score the quality of a mark scheme across five defined dimensions.  
# You must assign a score from 0-2 for each dimension according to the rubric below.

# Your response must be a single valid JSON object containing numeric scores and a total score.

# --------------------------------------------------------------------
# 1. Input:
# You will receive a JSON object containing:

# {
#     "question": "<the question>",
#     "mark_scheme": "<the generated mark scheme object with points and marks>",
#     "bloom_level": "<knowledge | understanding | application | analyze | evaluate | create>",
#     "difficulty": "<easy | medium | hard>",
#     "course_content": "<raw course text used to generate the question>"
# }

# --------------------------------------------------------------------
# 2. Output:
# You must output exactly one JSON object in the following format:

# {
#     "structural_appropriateness": 0 | 1 | 2,
#     "conceptual_coverage_quality": 0 | 1 | 2,
#     "precision_and_assessability": 0 | 1 | 2,
#     "cognitive_depth_alignment": 0 | 1 | 2,
#     "non_redundancy": 0 | 1 | 2
# }
# - Do NOT include explanations.
# - Do NOT include commentary.
# - Output only the JSON object.

# Example output:
# {
#     "structural_appropriateness": 2,
#     "conceptual_coverage_quality": 1,
#     "precision_and_assessability": 2,
#     "cognitive_depth_alignment": 1,
#     "non_redundancy": 0
# }

# --------------------------------------------------------------------
# 3. Scoring Dimensions and Rubric:

# (1) STRUCTURAL APPROPRIATENESS (0-2)
# Evaluate whether the structure (number of points, mark allocation, organization) is appropriate for the Bloom level.

# 0 = Clearly wrong structure (e.g., inappropriate number of points, incoherent mark allocation).
# 1 = Generally acceptable but uneven (minor structural imbalance or mild mismatch).
# 2 = Clean structure fully aligned with Bloom-level expectations.

# --------------------------------------------------------------------

# (2) CONCEPTUAL COVERAGE QUALITY (0-2)
# Evaluate whether the mark scheme captures the key conceptual components needed for a strong full-mark response.

# 0 = Missing major conceptual elements.
# 1 = Mostly complete but minor gaps or imbalance.
# 2 = Comprehensive, well-balanced coverage of essential ideas.

# --------------------------------------------------------------------

# (3) PRECISION AND ASSESSABILITY (0-2)
# Evaluate whether each point is specific, measurable, and clearly assessable.

# 0 = Vague, ambiguous, or unmarkable points.
# 1 = Mostly specific but some minor ambiguity.
# 2 = All points are clearly defined and assessable.

# --------------------------------------------------------------------

# (4) COGNITIVE DEPTH ALIGNMENT (0-2)
# Evaluate whether the mark scheme reflects the correct cognitive demand for the Bloom level.

# 0 = Clearly wrong cognitive depth (too shallow or too advanced).
# 1 = Mostly aligned but slight drift in depth.
# 2 = Fully aligned with required cognitive level.

# --------------------------------------------------------------------

# (5) NON-REDUNDANCY (0-2)
# Evaluate whether the points are distinct and non-overlapping.

# 0 = Repetitive or substantially overlapping points.
# 1 = Some minor conceptual overlap.
# 2 = Distinct, clearly differentiated ideas.

# --------------------------------------------------------------------
# 4. Scoring Rules

# - Base scores strictly on the content of the question, Bloom level, and mark scheme.
# - Do NOT assume missing information.
# - Do NOT reward verbosity.
# - Minor stylistic imperfections should not reduce scores.
# - Only reduce score when a clear substantive issue exists.
# - Scores must be integers (0, 1, or 2 only).

# --------------------------------------------------------------------
# 5. Output Formatting:
# - Always output a **single valid JSON object**.
# - Output only:
#     {
#         "structural_appropriateness": 0 | 1 | 2,
#         "conceptual_coverage_quality": 0 | 1 | 2,
#         "precision_and_assessability": 0 | 1 | 2,
#         "cognitive_depth_alignment": 0 | 1 | 2,
#         "non_redundancy": 0 | 1 | 2
#     }
# - Do NOT include commentary, analysis, or any text outside the JSON.
# - Ensure valid JSON syntax (no trailing commas, proper booleans, escaped characters).
# """