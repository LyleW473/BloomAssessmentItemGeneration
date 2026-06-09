# Assessment Question Generation (SAQ & SBQ)

Generate exam-style assessment questions from your own course material — complete with **mark schemes** and **expected answers** — then automatically **score and compare** the quality of what was generated.

Two question types are supported:

- **SAQ** — *Short-Answer Question*: a focused question answered in a few sentences.
- **SBQ** — *Scenario-Based Question*: a question wrapped in a short scenario the student must reason about.

Generation can optionally be guided by **Bloom's taxonomy** (so a question targets a
specific cognitive level — *remember, understand, apply, analyse, evaluate, create*), and
runs across several **generation pipelines** so you can compare strategies head-to-head.

There are two stages, run in order:

1. **Generate** — turn course content into questions (`scripts/generate_saqs.py`, `scripts/generate_sbqs.py`).
2. **Evaluate** — have an LLM judge score every generated question and rank the pipelines (`scripts/evaluate.py`).

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Add credentials
cp .env.example .env          # then edit .env (see "Credentials" below)

# 3. Make sure your course content exists at:
#    all_data/processed_assessment/contents/*.json   (see "Input data")

# 4. Generate questions
python scripts/generate_saqs.py
python scripts/generate_sbqs.py
#    -> all_data/synthetic/generated_assessment_student_questions/{saqs,sbqs}/

# 5. Score & compare the generated questions
python scripts/evaluate.py
#    -> evaluation_results/*.json

# 6. (optional) Plot the comparison
python scripts/generate_evaluation_plots.py
#    -> evaluation_results/*.pdf and *.png
```

> Run every script **from the project root** (e.g. `python scripts/evaluate.py`). Paths are
> anchored to the project root, so input/output always land in the right place regardless
> of where you launch from.

---

## Setup

### Requirements
- **Python 3.9+**
- Install dependencies: `pip install -r requirements.txt`

### Credentials
Copy `.env.example` to `.env` and fill in:

| Variable | What it is |
|---|---|
| `OPENAI_LLM_API_KEY` | API key for an OpenAI-compatible endpoint (the same endpoint also serves the Gemini models). |
| `OPENAI_BASE_URL` | Base URL of that endpoint. |

You need API access to the models the scripts request. The **evaluation** judge uses
`gpt-4o-mini`, `gpt-4.1-mini`, and `gemini-2.5-flash` by default — you can change this list
near the top of [scripts/evaluate.py](scripts/evaluate.py).

---

## Input data

Generation and evaluation both read **course content** from:

```
all_data/processed_assessment/contents/*.json     # one file per course / module
```

Each file holds an `extracted_texts` map — one entry per source file (lecture PDF, slides,
notes…), with the cleaned text and a classification. **Only entries classified as
`course_concept` are used** (so admin/overview files are skipped):

```json
{
  "extracted_texts": {
    "Week 1 - Introduction.pdf": {
      "file_path": "all_data/assessment/<course>/Week 1 - Introduction/IAI1.pdf",
      "compressed_text": "… cleaned course text that questions are grounded in …",
      "classification": { "category": "course_concept" }
    }
  }
}
```

> Data lives under `all_data/`, which is **git-ignored** — it holds your inputs and
> generated artifacts, not code.

---

## Stage 1 — Generation

```bash
python scripts/generate_saqs.py # short-answer questions
python scripts/generate_sbqs.py # scenario-based questions
```

Each script walks every course JSON and generates questions with **five strategies**, so you
can see which approach works best:

| Strategy | What it does |
|---|---|
| `zero_shot` | Ask the model directly, no extra guidance. |
| `zero_shot_bloom` | Same, but target a specific Bloom level. |
| `multi_stage_zero_shot` | Break generation into multiple steps. |
| `multi_stage_zero_shot_bloom` | Multi-step **and** Bloom-targeted. |
| `proposed_pipeline` | The full proposed approach (the one being evaluated against the rest). |

**Output** — one file per strategy × course, written incrementally so a long run is safe to
resume:

```
all_data/synthetic/generated_assessment_student_questions/{saqs,sbqs}/<strategy>___<course>.json
```

Each file is keyed by model name and contains the generated questions, their mark schemes,
and expected answers.

---

## Stage 2 — Evaluation

```bash
python scripts/evaluate.py
```

This is an **LLM-as-judge** pipeline. It does two things:

1. **Scores every question** against quality criteria (each on a 0–2 scale, normalised to 0–1):

   | Criterion | Checks that… |
   |---|---|
   | `bloom_level_alignment` | the cognitive demand matches the declared Bloom level |
   | `course_grounding` | the question genuinely depends on the course material |
   | `clarity` | the question is unambiguous and tightly scoped |
   | `single_task_integrity` | it asks for one coherent task, not several bundled together |
   | `objective_gradability_stem` | answers can be graded consistently from the question alone |
   | `objective_gradability_triple` | …and with the mark scheme + expected answer |
   | `mark_scheme_quality` | the mark scheme is structurally sound and factually accurate |
   | `answer_fidelity` | the expected answer actually covers the mark scheme |
   | `scenario_relevance_necessity` *(SBQ only)* | the scenario is necessary to answer the question |
   | `scenario_grounding` *(SBQ only)* | the scenario is accurate and course-grounded |

2. **Compares the pipelines head-to-head** using matched-task pairwise voting — the same
   questions, scored by multiple judge models — with bootstrap confidence intervals, broken
   down **per Bloom level** and **per judge model**.

**Output** — JSON files in `evaluation_results/` (one set per question type), e.g.:

```
evaluation_results/criterion_results_{saqs,sbqs}.json              # per-question scores
evaluation_results/criterion_results_metrics_{saqs,sbqs}.json      # aggregated metrics + yield + inter-rater agreement
evaluation_results/matched_task_results_{saqs,sbqs}.json           # pairwise pipeline comparison
evaluation_results/matched_task_samples_*_{saqs,sbqs}.json         # representative / contrasting examples (+ blind keys)
```

`evaluation_results/` is git-ignored (it's generated output).

### Plots (optional)

```bash
python scripts/generate_evaluation_plots.py
```

Renders the comparison as `main_figure.pdf/png` plus appendix charts (`appendix_stacked`,
`appendix_delta`) into `evaluation_results/`.

### Auxiliary utilities
- `scripts/recompute_mark_scheme_quality.py` — re-score just the mark-scheme-quality
  criterion without re-running the whole evaluation.
- `scripts/transform_discrimination_weights.py` — reshape the saved per-criterion
  discrimination weights.

---

## Project layout

```
BloomAssessmentItemGeneration/
├── scripts/                              # Entry points — run these
│   ├── set_path.py                       # Puts the project root on sys.path (imported by each script)
│   ├── generate_saqs.py                  # Generate SAQs
│   ├── generate_sbqs.py                  # Generate SBQs
│   ├── evaluate.py                       # Score & compare generated questions (SAQ & SBQ)
│   ├── generate_evaluation_plots.py      # Plots from evaluation results
│   ├── recompute_mark_scheme_quality.py  # Re-score just the mark-scheme criterion
│   └── transform_discrimination_weights.py
├── src/                                  # Source package
│   ├── globals.py                        # Data-directory / path constants
│   ├── llm_response_generation/          # LLM call + JSON-extraction helpers
│   ├── evaluation/assessment/            # Criterion verification + pairwise pipeline comparison
│   └── data_processing/synthetic/
│       ├── pipelines/                    # ZeroShot / MultiStage / Bloom generators (SAQ & SBQ)
│       ├── generation_and_verification/  # Generator + verifier systems
│       └── bloom_system_instructions/    # Prompt templates (SAQ, SBQ, shared)
└── all_data/                             # Inputs + generated artifacts (git-ignored)
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'dotenv'` / `openai` | Dependencies not installed — run `pip install -r requirements.txt`. |
| Auth / 401 errors from the model | `.env` missing or wrong — check `OPENAI_LLM_API_KEY` and `OPENAI_BASE_URL`. |
| `evaluate.py` finds no questions | Run generation first; confirm files exist under `all_data/synthetic/generated_assessment_student_questions/{saqs,sbqs}/`. |
| "Skipping file … classified as …" during a run | Expected — only `course_concept` entries are processed; other files are skipped by design. |
| Errors about a missing model (e.g. `gemini-2.5-flash`) | Your endpoint doesn't serve that model — edit the judge model list near the top of `scripts/evaluate.py`. |
| Output appears in an unexpected folder | Run scripts from the **project root** (`python scripts/...`). |