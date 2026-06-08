# Assessment Question Generation (SAQ & SBQ)
Short-answer question (SAQ) and scenario-based question (SBQ) generation pipelines.

- Given extracted course content, these scripts generate assessment questions, mark schemes, and expected answers, optionally guided by Bloom's taxonomy — across several generation pipelines and models.

## Layout

```
new_project/
├── scripts/                 # Entry points
│   ├── set_path.py          # Adds the project root to sys.path (imported by both scripts)
│   ├── generate_saqs_complete.py
│   └── generate_sbqs_bloom_complete.py
└── src/                     # Source package
    ├── globals.py           # Data-directory / path constants
    ├── llm_response_generation/      # LLM call + JSON-extraction helpers
    └── data_processing/synthetic/
        ├── pipelines/                # ZeroShot / MultiStage / Bloom generators (SAQ & SBQ)
        ├── generation_and_verification/  # Generator + verifier systems
        └── bloom_system_instructions/    # Prompt templates (SAQ, SBQ, shared)
```

## Setup

1. Python 3.8+.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure credentials — copy `.env.example` to `.env` and fill in:
   - `OPENAI_LLM_API_KEY` — key for the OpenAI-compatible endpoint (also serves Gemini models).
   - `OPENAI_BASE_URL` — base URL of that endpoint.

## Input / output data
Data is read/written to `all_data/`.

- **Input** (must exist): `all_data/processed_assessment/contents/*.json` — one JSON
  file per course/module, each containing an `extracted_texts` map of source files with
  their extracted content and classification metadata. Only files classified as
  `course_concept` are processed.
- **Output** (created automatically):
  `all_data/synthetic/generated_assessment_student_questions/{saqs,sbqs}/<generator>___<course>.json`.

## Running

From the project root:

```
python scripts/generate_saqs.py
python scripts/generate_sbqs.py
```

…or from inside the `scripts/` directory.

```
cd scripts
python generate_saqs.py
python generate_sbqs.py
```

Each script iterates over every course JSON and runs five generation strategies:
`zero_shot`, `zero_shot_bloom`, `multi_stage_zero_shot`, `multi_stage_zero_shot_bloom`,
and `proposed_pipeline` — across the configured models, writing results incrementally.