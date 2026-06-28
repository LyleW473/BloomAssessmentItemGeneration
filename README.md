# Assessment Question Generation (SAQ & SBQ)

Generate exam-style assessment questions from your own course material — complete with **mark schemes** and **expected answers** — then automatically **score and compare** the quality of what was generated.

Two question types are supported:

- **SAQ** — *Short-Answer Question*: a focused question answered in a few sentences.
- **SBQ** — *Scenario-Based Question*: a question wrapped in a short scenario the student must reason about.

Generation can optionally be guided by **Bloom's taxonomy** (so a question targets a
specific cognitive level — *remember, understand, apply, analyse, evaluate, create*), and
runs across several **generation pipelines** so you can compare strategies head-to-head.

The pipeline runs in stages:

0. **Preprocess** *(typical first step)* — turn raw course files (PDFs, slides, notes) into the cleaned `contents/*.json` that generation reads (`scripts/preprocess_data.py`).
1. **Generate** — turn that course content into questions (`scripts/generate_saqs.py`, `scripts/generate_sbqs.py`).
2. **Evaluate** — have an LLM judge score every generated question and rank the pipelines (`scripts/evaluate.py`).

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Add credentials
cp .env.example .env          # then edit .env (see "Credentials" below)

# 3. Preprocess raw course files -> the contents/*.json that generation reads
python scripts/preprocess_data.py ./path/to/course_folder --name my_course
#    -> all_data/processed_assessment/contents/my_course.json
#    (or supply your own contents/*.json — see "Input data")

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
| `OPENAI_LLM_API_KEY` | API key for an OpenAI-compatible endpoint (the same endpoint also serves the Gemini models). Works with OpenRouter too. |
| `OPENAI_BASE_URL` | Base URL of that endpoint — e.g. OpenAI `https://api.openai.com/v1`, or OpenRouter `https://openrouter.ai/api/v1`. |
| `LLAMA_CLOUD_API_KEY` | Llama Cloud key, used **only** by `scripts/preprocess_data.py` to parse non-`.txt` files (PDF/PPT/PPTX/DOCX/MD) via LlamaParse. Not needed for `.txt`-only preprocessing, generation, or evaluation. |

You need API access to the models the scripts request. The **evaluation** judge uses
`gpt-4o-mini`, `gpt-4.1-mini`, and `gemini-2.5-flash` by default — you can change this list
near the top of [scripts/evaluate.py](scripts/evaluate.py). **Preprocessing** model slugs are
passed per-run (see Stage 0); if your endpoint is OpenRouter, model names must be
provider-namespaced (e.g. `openai/gpt-4o`, `google/gemini-3.5-flash`).

---

## Input data

Generation and evaluation both read **course content** from:

```
all_data/processed_assessment/contents/*.json     # one file per course / module
```

These files are produced by `scripts/preprocess_data.py` (**Stage 0** below) — or you can
supply your own in the same shape. Each file holds an `extracted_texts` map — one entry per source file (lecture PDF, slides,
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

## Stage 0 — Preprocessing

Turn **raw course files** into the cleaned `contents/<name>.json` that generation reads. Point
the script at any file(s) and/or folder — it extracts the text, classifies each file, and
produces a compressed version optimised for question generation.

```bash
# A single file
python scripts/preprocess_data.py path/to/lecture.pdf

# A whole folder (every file inside, recursively) grouped into one module
python scripts/preprocess_data.py ./my_course/ --name intro_ml

# Several specific files into one named module
python scripts/preprocess_data.py a.pdf b.pptx notes.txt --name intro_ml
```

It runs **two phases**, saving incrementally so a failed run is cheap to resume (re-running
merges in new files without re-parsing the ones already done):

1. **Extract + classify** — pull text from each file and label it `course_concept`,
   `course_adjacent`, or `administrative`.
2. **Compress** — produce a high-fidelity `compressed_text` (guaranteed never larger than the
   original) that downstream generation is grounded in.

**Output** — one module file, plus a compression-stats metadata file:

```
all_data/processed_assessment/contents/<name>.json
all_data/processed_assessment/metadata/<name>_metadata.json
```

### Supported file types

| Type | How it's handled |
|---|---|
| `.txt` | Read directly — **no `LLAMA_CLOUD_API_KEY` needed**. |
| `.pdf`, `.ppt`, `.pptx`, `.docx`, `.md` | Parsed via **LlamaParse** — requires `LLAMA_CLOUD_API_KEY`. |
| `.mp4` | Skipped (no transcript parsing yet). |

### Options

| Flag | Purpose |
|---|---|
| `--name <name>` | Output module name → `contents/<name>.json`. Defaults to the single file/folder name; **required** when passing multiple loose files. |
| `--force-concept` | Treat every file as `course_concept` so nothing is filtered out downstream (see note). |
| `--overwrite` | Re-process from scratch instead of merging into an existing module (also re-runs compression). |
| `--classification-model <slug>` | Model for the classify step (default `openai/gpt-4o`). |
| `--compression-model <slug>` | Model for the compress step (default `google/gemini-3.5-flash`). |

> **Model slugs follow your endpoint.** The defaults are OpenRouter-style provider-namespaced
> slugs. If your `OPENAI_BASE_URL` points at plain OpenAI, pass bare names instead, e.g.
> `--classification-model gpt-4o --compression-model gpt-4o`.

> **Classification & filtering.** Generation only uses `course_concept` files. If a file you
> care about is labelled `course_adjacent`/`administrative`, the run warns and lists it in the
> end-of-run summary — re-run with `--force-concept` to include it.

---

## Stage 1 — Generation

```bash
python scripts/generate_saqs.py # short-answer questions
python scripts/generate_sbqs.py # scenario-based questions

# Optional: choose the models to benchmark across (provider-namespaced for OpenRouter)
python scripts/generate_saqs.py --models openai/gpt-4o-mini openai/gpt-4.1 google/gemini-2.5-flash
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

> **Reproducibility note — SBQ input text.** Earlier runs (including the published paper) had a
> bug that fed the **raw** extracted text to *every* SBQ generator instead of the
> `compressed_text` that SAQ uses. The bug was uniform across all SBQ pipelines, so the
> published *pipeline-vs-pipeline* comparison stays valid — but SBQ generation ran on
> uncompressed text. `generate_sbqs.py` now defaults to the **corrected** behavior
> (`--sbq-input compressed`); pass **`--sbq-input raw`** to reproduce the published paper exactly:
>
> ```bash
> python scripts/generate_sbqs.py --sbq-input raw   # reproduce the paper's SBQ behavior
> ```

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
│   ├── preprocess_data.py                # Stage 0: raw files -> contents/*.json (extract + classify + compress)
│   ├── extract_course_contents.py        # (legacy) extraction-only — superseded by preprocess_data.py
│   ├── generate_compressed_texts.py      # (legacy) compression-only — superseded by preprocess_data.py
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
│   └── data_processing/
│       ├── parser/                       # Document parsing (LlamaParse) — used by preprocessing
│       └── synthetic/
│           ├── utils.py                  # File-listing / week-ordering helpers
│           ├── pipelines/                # ZeroShot / MultiStage / Bloom generators (SAQ & SBQ)
│           ├── generation_and_verification/  # Generator + verifier systems
│           └── bloom_system_instructions/    # Prompt templates (SAQ, SBQ, shared, compression)
└── all_data/                             # Inputs + generated artifacts (git-ignored)
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'dotenv'` / `openai` | Dependencies not installed — run `pip install -r requirements.txt`. |
| Auth / 401 errors from the model | `.env` missing or wrong — check `OPENAI_LLM_API_KEY` and `OPENAI_BASE_URL`. |
| `LLAMA_CLOUD_API_KEY not found …` during preprocessing | A non-`.txt` file needs LlamaParse — set `LLAMA_CLOUD_API_KEY` in `.env`, or preprocess only `.txt` files. |
| `… is not a valid model ID` / 404 `No endpoints found` | Your endpoint doesn't serve that slug — pass a valid one via `--classification-model` / `--compression-model` (provider-namespaced for OpenRouter). |
| `Compressed (…) > raw (…); keeping raw extracted text instead` | Expected — the no-inflation guard fell back to the raw text for an already-dense file. |
| `evaluate.py` finds no questions | Run generation first; confirm files exist under `all_data/synthetic/generated_assessment_student_questions/{saqs,sbqs}/`. |
| "Skipping file … classified as …" during a run | Expected — only `course_concept` entries are processed; other files are skipped by design. |
| Errors about a missing model (e.g. `gemini-2.5-flash`) | Your endpoint doesn't serve that model — edit the judge model list near the top of `scripts/evaluate.py`. |
| Output appears in an unexpected folder | Run scripts from the **project root** (`python scripts/...`). |