import set_path
import os
import json
import random
import statistics
from itertools import combinations
from typing import List, Dict, Any
from src.evaluation.assessment.saq_evaluation import SAQEvaluator
from src.evaluation.assessment.helper_functions import (
    extract_pipeline_name_from_file_name,
    compute_criteria_pass_rates,
    aggregate_criterion_results_per_pipeline_global,
    compute_average_criterion_metrics_per_pipeline_global,
    aggregate_per_criterion_scores,
    calculate_discrimination_weights,
    calculate_weighted_pipeline_scores,
    compute_bloom_derived_metrics_per_pipeline,
    build_matched_task_mapping,
    compute_yield_metrics,
    aggregate_criterion_results_per_pipeline_model_global,
    compute_average_criterion_metrics_per_pipeline_model_global,
    compute_bloom_derived_metrics_per_pipeline_model,
    compute_criteria_pass_rates_by_model,
    compute_inter_rater_metrics,
    compute_bloom_level_agreement,
)
from src.evaluation.assessment.phases import (
    run_matched_task_pairwise_evaluation,
)

if __name__ == "__main__":

    from src.globals import SYNTHETIC_ASSESSMENT_DATA_QUESTIONS_DIR, BASE_EXTRACTED_QUESTIONS_DATA_DIR, PROJECT_ROOT

    # Outputs are written to a relative "evaluation_results/" path. Anchor the working
    # directory to the project root so the reports land in the same place regardless of
    # where the script is launched from (matching the PROJECT_ROOT-anchored globals).
    os.chdir(PROJECT_ROOT)

    random.seed(42)

    NUM_BOOTSTRAP_SAMPLES = 1000

    # Find original course content JSON files.
    module_json_files:List[str] = os.listdir(f"{BASE_EXTRACTED_QUESTIONS_DATA_DIR}/contents")
    print(f"Module JSON files found: {module_json_files}")
    module_json_paths = [f"{BASE_EXTRACTED_QUESTIONS_DATA_DIR}/contents/{file}" for file in module_json_files if file.endswith(".json")]

    # Create a mapping from 'course_name' -> 'file_name' -> 'extracted_text'
    file_path_to_extracted_text_mapping = {}
    for json_path in module_json_paths:
        with open(json_path, "r", encoding="utf-8") as f_json:
            module_data = json.load(f_json)
        
        extracted_texts_dict:Dict[str, Dict[str, str]] = module_data.get("extracted_texts", {})
        assert extracted_texts_dict is not None and isinstance(extracted_texts_dict, dict), "'extracted_texts' must be a dict" 

        for i, (file_name, content_dict) in enumerate(extracted_texts_dict.items()):
            print(f"Processing file {i+1}/{len(extracted_texts_dict)}: {file_name}")
            extracted_text = content_dict["compressed_text"]
            course_classification = content_dict.get("classification", {})

            # Skip files that are not classified as "course_concept"
            # print(f"Course classification: {json.dumps(course_classification, indent=4)}")
            if course_classification.get("category", "") != "course_concept":
                print(f"Skipping file {file_name} as it is classified as '{course_classification.get('category', 'unknown')}'")
                continue
            
            full_file_path = content_dict["file_path"] # Same key used for generated questions
            file_path_to_extracted_text_mapping[full_file_path] = extracted_text
    
    print(f"Number of files with extracted text: {len(file_path_to_extracted_text_mapping)}")

    evaluator = SAQEvaluator()

    # Get criterion results for every generated question.
    from src.evaluation.assessment.criterion_verifier import CriterionVerifier
    os.makedirs("evaluation_results", exist_ok=True)

    # Initialise the CriterionVerifier with the desired models for evaluation
    criterion_verifier = CriterionVerifier(
        client=evaluator.client,
        model_names=[
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gemini-2.5-flash",
        ]
    )
    question_types_to_dir = {
        "saqs": f"{SYNTHETIC_ASSESSMENT_DATA_QUESTIONS_DIR}/saqs",
        "sbqs": f"{SYNTHETIC_ASSESSMENT_DATA_QUESTIONS_DIR}/sbqs",
    }

    all_pipeline_names = set()
    # Accumulated mapping across all question types: file -> pipeline -> model -> list[questions]
    file_to_pipeline_to_model_to_saqs_mapping = {}
    # Per-type mapping for type-specific per-file pairwise evaluation
    per_type_file_to_pipeline_to_model_to_saqs_mapping = {}
    for question_type, generated_dir in question_types_to_dir.items():
        if not os.path.exists(generated_dir):
            print(f"Skipping '{question_type}' because directory does not exist: {generated_dir}")
            continue

        print(f"\n{'='*80}")
        print(f"Generating criterion results for question type: {question_type}")
        print(f"Source directory: {generated_dir}")
        print(f"{'='*80}\n")

        all_json_files = [f for f in sorted(os.listdir(generated_dir)) if f.endswith(".json")]
        print(f"Found {len(all_json_files)} json files for '{question_type}'")

        # Create mapping: file -> pipeline -> model -> list[questions]
        file_to_pipeline_to_model_to_questions_mapping = {}
        for json_file in all_json_files:
            json_path = f"{generated_dir}/{json_file}"
            pipeline_name = extract_pipeline_name_from_file_name(json_file)
            print(f"Extracted pipeline name: {pipeline_name} from file name: {json_file}")

            with open(json_path, "r", encoding="utf-8") as f_json:
                all_packed_data = json.load(f_json)
                print(f"Loaded {len(all_packed_data)} models from {json_file}")

                for model_name, questions in all_packed_data.items():
                    for packed_data in questions:
                        file_name = packed_data["file_path"]
                        if file_name not in file_to_pipeline_to_model_to_questions_mapping:
                            file_to_pipeline_to_model_to_questions_mapping[file_name] = {}
                        if pipeline_name not in file_to_pipeline_to_model_to_questions_mapping[file_name]:
                            file_to_pipeline_to_model_to_questions_mapping[file_name][pipeline_name] = {}
                        if model_name not in file_to_pipeline_to_model_to_questions_mapping[file_name][pipeline_name]:
                            file_to_pipeline_to_model_to_questions_mapping[file_name][pipeline_name][model_name] = []
                        file_to_pipeline_to_model_to_questions_mapping[file_name][pipeline_name][model_name].append(packed_data)

        for file_name, pipeline_to_model_to_questions in file_to_pipeline_to_model_to_questions_mapping.items():
            print(f"File: {file_name}")
            print(f"Pipelines found: {list(pipeline_to_model_to_questions.keys())}")
            for pipeline_name, model_to_questions in pipeline_to_model_to_questions.items():
                print(f"Pipeline: {pipeline_name} | Models: {list(model_to_questions.keys())}")
                for model_name, questions in model_to_questions.items():
                    print(f"  Model: {model_name} | Num questions: {len(questions)}")

        # Accumulate into the global mapping across all question types
        for file_name, p_to_m_to_q in file_to_pipeline_to_model_to_questions_mapping.items():
            if file_name not in file_to_pipeline_to_model_to_saqs_mapping:
                file_to_pipeline_to_model_to_saqs_mapping[file_name] = {}
            for pipeline_name, m_to_q in p_to_m_to_q.items():
                if pipeline_name not in file_to_pipeline_to_model_to_saqs_mapping[file_name]:
                    file_to_pipeline_to_model_to_saqs_mapping[file_name][pipeline_name] = {}
                for model_name, qs in m_to_q.items():
                    file_to_pipeline_to_model_to_saqs_mapping[file_name][pipeline_name].setdefault(model_name, []).extend(qs)

        # Save per-type mapping for type-specific per-file pairwise evaluation
        per_type_file_to_pipeline_to_model_to_saqs_mapping[question_type] = file_to_pipeline_to_model_to_questions_mapping

        num_to_process = sum(
            len(questions)
            for pipeline_to_model_to_questions in file_to_pipeline_to_model_to_questions_mapping.values()
            for model_to_questions in pipeline_to_model_to_questions.values()
            for questions in model_to_questions.values()
        )

        output_path = f"evaluation_results/criterion_results_{question_type}.json"
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                all_criterion_results = json.load(f)
                print(f"Loaded existing criterion results from disk: {output_path}")
        else:
            all_criterion_results = {}
            print(f"No existing criterion results found at {output_path}. Starting fresh.")

        # Count questions where ALL currently-configured evaluator models have already scored.
        # This makes the skip logic model-aware: adding a new model to model_names causes previously-cached questions to be re-detected as needing processing.
        current_eval_models = set(criterion_verifier.model_names)
        num_already_processed = 0
        for file_path in all_criterion_results:
            for pipeline_name in all_criterion_results[file_path]:
                for gm in all_criterion_results[file_path][pipeline_name]:
                    for result in all_criterion_results[file_path][pipeline_name][gm]:
                        scored_models = set(next(iter(result.get("raw_scores", {}).values()), {}).keys())
                        if current_eval_models.issubset(scored_models):
                            num_already_processed += 1

        print(f"Number of '{question_type}' questions already processed (all {len(current_eval_models)} evaluator model(s)): {num_already_processed}/{num_to_process}")

        if num_already_processed >= num_to_process:
            print(f"All '{question_type}' questions have already been processed. Skipping criterion verification.")
            for file_path in all_criterion_results:
                all_pipeline_names.update(all_criterion_results[file_path].keys())
            continue

        print(f"Processing {num_to_process - num_already_processed} remaining '{question_type}' questions.")

        global_question_counter = 0
        for file_path, pipeline_to_model_to_questions in file_to_pipeline_to_model_to_questions_mapping.items():
            for pipeline_name, model_to_questions in pipeline_to_model_to_questions.items():
                for model_name, questions in model_to_questions.items():
                    print(f"\nPerforming criterion verification for type={question_type} | file={file_path} | pipeline={pipeline_name} | model={model_name}")

                    all_pipeline_names.add(pipeline_name)

                    if file_path not in all_criterion_results:
                        all_criterion_results[file_path] = {}
                    if pipeline_name not in all_criterion_results[file_path]:
                        all_criterion_results[file_path][pipeline_name] = {}
                    if model_name not in all_criterion_results[file_path][pipeline_name]:
                        all_criterion_results[file_path][pipeline_name][model_name] = []

                    num_existing_results = len(all_criterion_results[file_path][pipeline_name][model_name])

                    for question_idx, question_dict in enumerate(questions):
                        global_question_counter += 1

                        if question_idx < num_existing_results:
                            existing_result = all_criterion_results[file_path][pipeline_name][model_name][question_idx]
                            scored_models = set(next(iter(existing_result.get("raw_scores", {}).values()), {}).keys())
                            missing_models = [m for m in criterion_verifier.model_names if m not in scored_models]

                            if not missing_models:
                                print(
                                    f"[{question_type.upper()} Global {global_question_counter}/{num_to_process}] "
                                    f"Question {question_idx + 1}/{len(questions)} already processed (all evaluator models). Skipping."
                                )
                                continue

                            # Some evaluator models are new, update in place without re-running existing models
                            print(
                                f"[{question_type.upper()} Global {global_question_counter}/{num_to_process}] "
                                f"Question {question_idx + 1}/{len(questions)}: adding {len(missing_models)} new evaluator model(s): {missing_models}"
                            )
                            extracted_text = file_path_to_extracted_text_mapping[file_path]
                            updated_result = criterion_verifier.update_result_with_new_models(
                                existing_result=existing_result,
                                generated_question_dict=question_dict,
                                extracted_text=extracted_text,
                                new_model_names=missing_models,
                                max_retries=3
                            )
                            all_criterion_results[file_path][pipeline_name][model_name][question_idx] = updated_result
                            with open(output_path, "w", encoding="utf-8") as f:
                                json.dump(all_criterion_results, f, indent=4)
                            print("✓ Updated criterion result with new evaluator models")
                            continue

                        print(
                            f"[{question_type.upper()} Global {global_question_counter}/{num_to_process}] "
                            f"Processing question {question_idx + 1}/{len(questions)}"
                        )

                        extracted_text = file_path_to_extracted_text_mapping[file_path]

                        max_retries = 3
                        retry_delay = 2
                        criterion_results = None

                        for retry_attempt in range(max_retries):
                            try:
                                criterion_results = criterion_verifier.perform_checks(
                                    generated_question_dict=question_dict,
                                    extracted_text=extracted_text,
                                    max_retries=3
                                )
                                print("✓ Completed criterion verification")
                                break
                            except (RuntimeError, Exception) as e:
                                print(f"Criterion verification failed (attempt {retry_attempt + 1}/{max_retries}): {e}")
                                if retry_attempt < max_retries - 1:
                                    import time
                                    print(f"Waiting {retry_delay} seconds before retry...")
                                    time.sleep(retry_delay)
                                    retry_delay *= 2
                                else:
                                    print(f"Exiting script after {max_retries} failed attempts for this question.")
                                    raise e

                        if criterion_results is not None:
                            criterion_results["metadata"] = {
                                "question_type": question_type,
                                "file_path": file_path,
                                "pipeline_name": pipeline_name,
                                "generator_model_name": model_name,
                                "question_index": question_idx,
                                "generated_triple": question_dict
                            }
                            all_criterion_results[file_path][pipeline_name][model_name].append(criterion_results)

                            with open(output_path, "w", encoding="utf-8") as f:
                                json.dump(all_criterion_results, f, indent=4)

    pipeline_names = sorted(list(all_pipeline_names))

    # Load per-type criterion results
    per_type_criterion_results = {}
    for qt in question_types_to_dir.keys():
        qt_output_path = f"evaluation_results/criterion_results_{qt}.json"
        if os.path.exists(qt_output_path):
            with open(qt_output_path, "r", encoding="utf-8") as f:
                per_type_criterion_results[qt] = json.load(f)

    pairs = list(combinations(pipeline_names, 2))
    print(f"Pipeline pairs for evaluation: {pairs}")

    # ========================================================================================
    # FULL EVALUATION PIPELINE — RUN SEPARATELY FOR EACH QUESTION TYPE
    # ========================================================================================

    for question_type, qt_cr in per_type_criterion_results.items():
        qt_saqs_mapping = per_type_file_to_pipeline_to_model_to_saqs_mapping.get(question_type)
        if qt_saqs_mapping is None:
            print(f"Skipping all evaluations for '{question_type}' (no question mapping).")
            continue

        if question_type == "saqs":
            criteria = evaluator.TIE_BREAK_PRIORITY_SAQ.copy()
        elif question_type == "sbqs":
            criteria = evaluator.TIE_BREAK_PRIORITY_SBQ.copy()
        else:
            raise ValueError(f"Unsupported question type: {question_type}")

        print(f"\n{'='*80}")
        print(f"EVALUATING QUESTION TYPE: {question_type.upper()}")
        print(f"{'='*80}\n")

        # Flatten criterion results to file -> pipeline -> list (for criterion analysis helpers)
        qt_file_to_pipeline_to_criterion_results_mapping = {}
        for file_name, pipeline_to_model_to_saqs in qt_saqs_mapping.items():
            qt_file_to_pipeline_to_criterion_results_mapping[file_name] = {}
            for pipeline_name, model_to_saqs in pipeline_to_model_to_saqs.items():
                flattened_results = []
                model_to_results = qt_cr.get(file_name, {}).get(pipeline_name, {})
                for model_name in model_to_saqs.keys():
                    flattened_results.extend(model_to_results.get(model_name, []))
                qt_file_to_pipeline_to_criterion_results_mapping[file_name][pipeline_name] = flattened_results

        # --- CRITERIA PASS RATES ---

        per_file_criteria_pass_rates = compute_criteria_pass_rates(
            pipeline_names=pipeline_names,
            all_criterion_results=qt_file_to_pipeline_to_criterion_results_mapping,
            criteria=criteria
        )
        print(f"\n{'='*80}")
        print(f"CRITERIA PASS RATES — {question_type.upper()}")
        print(f"{'='*80}\n")
        for pipeline_name in pipeline_names:
            print(f"\n{pipeline_name}:")
            print(f"Overall: {per_file_criteria_pass_rates[pipeline_name]['overall_pass_rate']:.1%}")
            for criterion, rate in per_file_criteria_pass_rates[pipeline_name]['per_criterion'].items():
                print(f"  - {criterion}: {rate:.1%}")

        # --- CRITERION METRICS & DISCRIMINATION WEIGHTS ---

        aggregated_criterion_results_per_pipeline_global = aggregate_criterion_results_per_pipeline_global(
            all_criterion_results=qt_file_to_pipeline_to_criterion_results_mapping
        )
        for pipeline_name, criterion_results in aggregated_criterion_results_per_pipeline_global.items():
            print(f"Pipeline: {pipeline_name} | Total criterion results aggregated: {len(criterion_results)}")

        average_criterion_metrics_per_pipeline_global = compute_average_criterion_metrics_per_pipeline_global(
            aggregated_criterion_results_per_pipeline=aggregated_criterion_results_per_pipeline_global,
            criteria=criteria
        )

        print(f"\n{'='*80}")
        print(f"AVERAGE CRITERION METRICS PER PIPELINE — {question_type.upper()}")
        print(f"{'='*80}\n")
        print(json.dumps(average_criterion_metrics_per_pipeline_global, indent=4))

        print(f"\n{'='*80}")
        print(f"AVERAGE CRITERION METRICS WITH MARGIN RANGES — {question_type.upper()}")
        print(f"{'='*80}\n")
        for criterion in criteria:
            print(f"\n{criterion}:")
            criterion_means = {}
            for pipeline_name in pipeline_names:
                mean_score = average_criterion_metrics_per_pipeline_global[pipeline_name].get(criterion, 0)
                criterion_means[pipeline_name] = mean_score
                print(f"  {pipeline_name}: {mean_score:.4f}")
            max_mean = max(criterion_means.values())
            min_mean = min(criterion_means.values())
            print(f"  Range: {min_mean:.4f} - {max_mean:.4f} (Δ={max_mean - min_mean:.4f})")

        pipeline_rankings = []
        for pipeline_name in pipeline_names:
            criterion_means = average_criterion_metrics_per_pipeline_global[pipeline_name]
            overall_average_score = sum(criterion_means.get(criterion, 0) for criterion in criteria) / len(criteria)
            pipeline_rankings.append((pipeline_name, overall_average_score))
        pipeline_rankings.sort(key=lambda x: x[1], reverse=True)

        print(f"\n{'='*80}")
        print(f"DISCRIMINATION-WEIGHTED PIPELINE RANKINGS — {question_type.upper()}")
        print(f"{'='*80}\n")

        per_criterion_scores = aggregate_per_criterion_scores(
            pipeline_names=pipeline_names,
            criteria=criteria,
            aggregated_criterion_results_per_pipeline_global=aggregated_criterion_results_per_pipeline_global
        )

        discrimination_power, discrimination_weights = calculate_discrimination_weights(
            pipeline_names=pipeline_names,
            criteria=criteria,
            per_criterion_scores=per_criterion_scores,
            verbose=True
        )

        print("Discrimination-based criterion weights (square root transformed):")
        for criterion in sorted(discrimination_weights.keys(), key=lambda x: discrimination_weights[x], reverse=True):
            print(f"  {criterion}: {discrimination_weights[criterion]:.4f} (raw power: {discrimination_power[criterion]:.4f})")

        weighted_pipeline_scores = calculate_weighted_pipeline_scores(
            pipeline_names=pipeline_names,
            criteria=criteria,
            per_criterion_scores=per_criterion_scores,
            discrimination_weights=discrimination_weights
        )

        print(f"\nDiscrimination-weighted pipeline scores:")
        weighted_ranked_pipelines = sorted(weighted_pipeline_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (pipeline, score) in enumerate(weighted_ranked_pipelines, 1):
            print(f" {rank}. {pipeline}: {score:.4f}")

        print(f"\nDiscrimination-weighted margins:")
        for i in range(len(weighted_ranked_pipelines) - 1):
            pipeline_a, score_a = weighted_ranked_pipelines[i]
            pipeline_b, score_b = weighted_ranked_pipelines[i + 1]
            margin = score_a - score_b
            print(f"  {pipeline_a} vs {pipeline_b}: +{margin:.4f}")

        bloom_derived_metrics = compute_bloom_derived_metrics_per_pipeline(
            aggregated_criterion_results_per_pipeline=aggregated_criterion_results_per_pipeline_global
        )

        print(f"\n{'='*80}")
        print(f"BLOOM-DERIVED METRICS PER PIPELINE — {question_type.upper()}")
        print(f"{'='*80}\n")
        for pipeline_name in pipeline_names:
            metrics = bloom_derived_metrics[pipeline_name]
            print(f"{pipeline_name}:")
            print(f"  Bloom alignment accuracy: {metrics['bloom_alignment_accuracy']:.1%}")
            print(f"  Higher-order success rate: {metrics['higher_order_success_rate']:.1%}")

        # --- PER (PIPELINE, MODEL) CRITERION METRICS ---
        aggregated_per_pipeline_model = aggregate_criterion_results_per_pipeline_model_global(
            all_criterion_results=qt_cr
        )

        average_criterion_metrics_per_pipeline_model = compute_average_criterion_metrics_per_pipeline_model_global(
            aggregated_per_pipeline_model=aggregated_per_pipeline_model,
            criteria=criteria
        )

        bloom_derived_metrics_per_model = compute_bloom_derived_metrics_per_pipeline_model(
            aggregated_per_pipeline_model=aggregated_per_pipeline_model
        )

        criteria_pass_rates_by_model = compute_criteria_pass_rates_by_model(
            pipeline_names=pipeline_names,
            all_criterion_results=qt_cr,
            criteria=criteria
        )

        qt_criterion_metrics_path = f"evaluation_results/criterion_results_metrics_{question_type}.json"
        with open(qt_criterion_metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "base": {
                    "results": average_criterion_metrics_per_pipeline_global,
                    "rankings": pipeline_rankings,
                    "bloom_derived": bloom_derived_metrics
                },
                "discrimination_weighted": {
                    "discrimination_power": discrimination_power,
                    "discrimination_weights": discrimination_weights,
                    "weighted_pipeline_scores": weighted_pipeline_scores,
                    "rankings": weighted_ranked_pipelines
                },
                "criteria_pass_rates": per_file_criteria_pass_rates,
                "per_model": {
                    "base": {
                        "results": average_criterion_metrics_per_pipeline_model,
                        "bloom_derived": bloom_derived_metrics_per_model
                    },
                    "criteria_pass_rates": criteria_pass_rates_by_model
                }
            }, f, indent=4)
        print(f"\nDiscrimination weights saved to: {qt_criterion_metrics_path}")

        # ── YIELD METRICS (systems metric, reported separately from quality) ──

        print(f"\n{'='*80}")
        print(f"YIELD METRICS (SYSTEMS METRIC, NOT QUALITY) — {question_type.upper()}")
        print(f"{'='*80}\n")
        print("Yield measures operational capability: what proportion of task slots produced")
        print("at least one valid question (discrimination-weighted score >= 0.5)?\n")

        # Build the matched-task map once — reused by both yield and matched evaluation
        task_map = build_matched_task_mapping(
            file_to_pipeline_to_model_to_saqs_mapping=qt_saqs_mapping,
            all_criterion_results=qt_cr
        )
        print(f"Matched task mapping: {len(task_map)} task slots total")
        for diff in ["easy", "medium", "hard", "unknown"]:
            diff_count = sum(1 for tk in task_map if tk[1] == diff)
            if diff_count > 0:
                print(f"  {diff}: {diff_count} task slots")

        yield_metrics = compute_yield_metrics(
            pipeline_names=pipeline_names,
            task_mapping=task_map,
            discrimination_weights=discrimination_weights,
            validity_threshold=0.5,
        )
        for pipeline_name in pipeline_names:
            ym = yield_metrics[pipeline_name]
            print(f"\n{pipeline_name}:")
            print(f"  Task yield rate:      {ym['task_yield_rate']:.1%}  "
                  f"({ym['total_task_slots']} task slots total)")
            print(f"  Mean valid/task:      {ym['mean_valid_per_task']:.2f}")
            print(f"  Total valid items:    {ym['total_valid']} / {ym['total_questions']}")

        # ── MATCHED-TASK SAMPLES EXPORT ──────────────────────────────────────
        # REPRESENTATIVE: one median triple per difficulty (typical, non-cherry-picked).
        # ILLUSTRATIVE:   two contrast examples per difficulty:
        #   max_score_gap — proposed - min(all others) overall criterion score gap
        #   bloom_bla_gap — proposed_bla - min(all others bla) Bloom alignment gap
        # Each output file is written in BLIND form: pipeline keys are replaced with
        # randomly assigned labels (pipeline_A … pipeline_E, shuffled independently per
        # entry) and file_path / generation_model are stripped. Labels are sorted
        # lexicographically in the output. A companion *_key.json maps labels back to
        # real identifiers.

        print(f"\n{'='*80}")
        print(f"MATCHED-TASK SAMPLES EXPORT — {question_type.upper()}")
        print(f"{'='*80}\n")

        _total_w = sum(discrimination_weights.values()) or 1.0

        def _weighted_score_for_crit(crit: Dict[str, Any]) -> float:
            return sum(crit.get("metrics", {}).get(k, 0.0) * w for k, w in discrimination_weights.items()) / _total_w

        def _median_idx(scores: list) -> int:
            """Return the index of the element closest to the median of scores."""
            if not scores:
                return 0
            sorted_scores = sorted(scores)
            median_val = sorted_scores[len(sorted_scores) // 2]  # lower-median
            return min(range(len(scores)), key=lambda i: abs(scores[i] - median_val))

        # Resolve pipeline alias for illustrative axes
        _proposed = next((p for p in pipeline_names if "proposed" in p), None)

        # First pass: collect all candidates per difficulty
        candidates_per_difficulty: Dict[str, list] = {}
        for task_key in sorted(task_map.keys()):
            file_path, difficulty = task_key
            p_to_m_to_data = task_map[task_key]

            # Only keep triples where every pipeline has data
            if not all(p in p_to_m_to_data for p in pipeline_names):
                continue

            model_sets = [set(p_to_m_to_data[p].keys()) for p in pipeline_names]
            common_models = sorted(set.intersection(*model_sets))

            for model_name in common_models:
                pipeline_outputs = {}
                pipeline_median_scores = []
                for pipeline_name in pipeline_names:
                    slot = p_to_m_to_data[pipeline_name][model_name]
                    scores = [_weighted_score_for_crit(c) for c in slot["criteria"]]
                    med_idx = _median_idx(scores)
                    med_q = slot["questions"][med_idx]
                    med_crit = slot["criteria"][med_idx]
                    med_score = scores[med_idx]
                    pipeline_median_scores.append(med_score)

                    bloom_details = med_crit.get("bloom_alignment_details", {})
                    declared_bloom_level = med_q.get("bloom_level") or bloom_details.get("declared_bloom_level")
                    true_bloom_level = bloom_details.get("majority_true_bloom_level")
                    entry = {
                        "question": med_q.get("question"),
                        "declared_bloom_level": declared_bloom_level,
                        "true_bloom_level": true_bloom_level,
                        "mark_scheme": med_q.get("mark_scheme"),
                        "total_marks": med_q.get("total_marks"),
                        "expected_answer": med_q.get("expected_answer"),
                        "criterion_score": round(med_score, 4),
                        "criterion_scores": {k: round(v, 4) for k, v in med_crit.get("metrics", {}).items()},
                    }
                    if question_type == "sbqs":
                        entry["scenario"] = med_q.get("scenario")
                    pipeline_outputs[pipeline_name] = entry

                triple_score = sum(pipeline_median_scores) / len(pipeline_median_scores)
                candidates_per_difficulty.setdefault(difficulty, []).append((triple_score, {
                    "file_path": file_path,
                    "generation_model": model_name,
                    "mean_criterion_score": round(triple_score, 4),
                    "pipelines": pipeline_outputs,
                }))

        # ── Representative: median triple per difficulty ──────────────────────
        matched_samples: Dict[str, Any] = {}
        for diff in ["easy", "medium", "hard"]:
            candidates = candidates_per_difficulty.get(diff, [])
            if not candidates:
                continue
            all_triple_scores = [c[0] for c in candidates]
            rep_idx = _median_idx(all_triple_scores)
            rep_score = all_triple_scores[rep_idx]
            n_cand = len(all_triple_scores)
            entry = candidates[rep_idx][1]
            entry["candidate_pool_stats"] = {
                "n_candidates": n_cand,
                "score_distribution": {
                    "mean": round(sum(all_triple_scores) / n_cand, 4),
                    "std": round(statistics.pstdev(all_triple_scores), 4),
                    "min": round(min(all_triple_scores), 4),
                    "max": round(max(all_triple_scores), 4),
                },
                # Fraction of candidates with a strictly lower mean_criterion_score;
                # ~50 confirms the selection is genuinely middle-of-the-pack.
                "selected_percentile": round(
                    sum(1 for s in all_triple_scores if s < rep_score) / n_cand * 100, 1
                ),
            }
            matched_samples[diff] = entry

        print(f"Representative samples selected per difficulty (median triple score):")
        for diff, entry in matched_samples.items():
            print(f"  {diff}: {entry['file_path']} | model={entry['generation_model']} | score={entry['mean_criterion_score']:.4f}")

        # ── Blind export: anonymise pipeline labels, strip identity fields ───
        blind_rep: Dict[str, Any] = {}
        rep_key: Dict[str, Any] = {}
        for diff, entry in matched_samples.items():
            _shuffled = list(pipeline_names)
            random.shuffle(_shuffled)
            _label_map = {name: f"pipeline_{chr(65 + i)}" for i, name in enumerate(_shuffled)}
            blind_entry = {k: v for k, v in entry.items() if k not in ("file_path", "generation_model")}
            blind_entry["pipelines"] = dict(sorted(
                {_label_map[p]: v for p, v in entry["pipelines"].items() if p in _label_map}.items()
            ))
            blind_rep[diff] = blind_entry
            rep_key[diff] = {
                "file_path": entry["file_path"],
                "generation_model": entry["generation_model"],
                # label -> real pipeline name
                "pipeline_label_map": {v: k for k, v in _label_map.items()},
            }

        qt_representative_path = f"evaluation_results/matched_task_samples_representative_{question_type}.json"
        with open(qt_representative_path, "w", encoding="utf-8") as f:
            json.dump(blind_rep, f, indent=4)
        qt_representative_key_path = qt_representative_path.replace(".json", "_key.json")
        with open(qt_representative_key_path, "w", encoding="utf-8") as f:
            json.dump(rep_key, f, indent=4)
        print(f"Representative samples (blind) saved to:  {qt_representative_path}")
        print(f"Representative samples key saved to:      {qt_representative_key_path}")

        # ── Illustrative contrast ─────────────────────────────────────────────
        # axis 1 — max_score_gap:  triple that maximises proposed_pipeline overall score gap over its
        #                          worst competitor (proposed - min(other) across all baselines)
        # axis 2 — bloom_bla_gap:  triple that maximises proposed_pipeline BLA advantage over its
        #                          worst competitor (proposed_bla - min(other_bla) across all baselines)
        illustrative_samples: Dict[str, Any] = {}
        for diff in ["easy", "medium", "hard"]:
            candidates = candidates_per_difficulty.get(diff, [])
            if not candidates:
                continue
            diff_illustrative: Dict[str, Any] = {}

            # Axis 1: max overall score gap — proposed.criterion_score - min(all other pipelines)
            # Records which pipeline was the worst baseline on this specific task.
            if _proposed:
                _ax1_others = [p for p in pipeline_names if p != _proposed]

                def _axis1_gap(c):
                    pipelines = c[1]["pipelines"]
                    proposed_score = pipelines.get(_proposed, {}).get("criterion_score", 0.0)
                    other_scores = [
                        pipelines.get(p, {}).get("criterion_score", 0.0)
                        for p in _ax1_others if p in pipelines
                    ]
                    return proposed_score - min(other_scores) if other_scores else 0.0

                best_gap_c = max(candidates, key=_axis1_gap)
                gap = _axis1_gap(best_gap_c)
                _all_ax1_gaps = sorted([_axis1_gap(c) for c in candidates])
                ax1_gap_percentile = round(
                    sum(1 for g in _all_ax1_gaps if g < gap) / len(_all_ax1_gaps) * 100, 1
                )
                ax1_runner_up = round(sorted(_all_ax1_gaps, reverse=True)[1], 4) if len(_all_ax1_gaps) > 1 else None
                pipelines_ax1 = best_gap_c[1]["pipelines"]
                _ax1_best_overall = max(
                    pipelines_ax1.get(p, {}).get("criterion_score", 0.0) for p in pipeline_names
                )
                ax1_entry = dict(best_gap_c[1])
                ax1_entry["illustrative_note"] = {
                    "axis": "max_score_gap",
                    "mean_criterion_score": ax1_entry.pop("mean_criterion_score", None),
                    "best_overall_score": round(_ax1_best_overall, 4),
                    "score_gap": round(gap, 4),
                    # Percentile rank of this gap among all candidates in this difficulty bucket;
                    # high values confirm this is a genuinely extreme example, not a typical one.
                    "gap_percentile_rank": ax1_gap_percentile,
                    # Second-best gap — shows how dominant the selected example is.
                    "runner_up_gap": ax1_runner_up,
                }
                diff_illustrative["max_score_gap"] = ax1_entry

            # Axis 2: max BLA gap — proposed_bla - min(other_bla) over all non-proposed pipelines.
            # Selects the task where proposed_pipeline has the largest hierarchical Bloom
            # alignment advantage over its worst competitor on this specific task.
            if _proposed:
                _other_pipelines = [p for p in pipeline_names if p != _proposed]

                def _bla_gap(c):
                    pipelines = c[1]["pipelines"]
                    proposed_bla = pipelines.get(_proposed, {}).get("criterion_scores", {}).get("bloom_level_alignment", 0.0)
                    other_blas = [
                        pipelines.get(p, {}).get("criterion_scores", {}).get("bloom_level_alignment", 0.0)
                        for p in _other_pipelines
                        if p in pipelines
                    ]
                    if not other_blas:
                        return 0.0
                    return proposed_bla - min(other_blas)

                best_bloom_c = max(
                    candidates,
                    key=lambda c: (
                        _bla_gap(c),
                        # Tie-break: largest axis-1 overall score gap
                        c[1]["pipelines"].get(_proposed, {}).get("criterion_score", 0.0)
                        - min(
                            c[1]["pipelines"].get(p, {}).get("criterion_score", 0.0)
                            for p in _other_pipelines
                            if p in c[1]["pipelines"]
                        ) if _other_pipelines else 0.0
                    )
                )
                bla_gap_val = _bla_gap(best_bloom_c)
                _all_bla_gaps = [_bla_gap(c) for c in candidates]
                bla_gap_percentile = round(
                    sum(1 for g in _all_bla_gaps if g < bla_gap_val) / len(_all_bla_gaps) * 100, 1
                )
                bla_runner_up = round(sorted(_all_bla_gaps, reverse=True)[1], 4) if len(_all_bla_gaps) > 1 else None
                pipelines_ax2 = best_bloom_c[1]["pipelines"]
                proposed_bla = pipelines_ax2.get(_proposed, {}).get("criterion_scores", {}).get("bloom_level_alignment", 0.0)
                worst_baseline = min(
                    _other_pipelines,
                    key=lambda p: pipelines_ax2.get(p, {}).get("criterion_scores", {}).get("bloom_level_alignment", float("inf"))
                ) if _other_pipelines else None
                worst_baseline_bla = pipelines_ax2.get(worst_baseline, {}).get("criterion_scores", {}).get("bloom_level_alignment", 0.0) if worst_baseline else None

                ax2_entry = dict(best_bloom_c[1])
                _ax2_best_overall = max(
                    pipelines_ax2.get(p, {}).get("criterion_score", 0.0) for p in pipeline_names
                )
                ax2_entry["illustrative_note"] = {
                    "axis": "bloom_bla_gap",
                    "mean_criterion_score": ax2_entry.pop("mean_criterion_score", None),
                    "best_overall_score": round(_ax2_best_overall, 4),
                    "bloom_bla_gap": round(bla_gap_val, 4),
                    # Percentile rank of this BLA gap in the difficulty bucket.
                    "gap_percentile_rank": bla_gap_percentile,
                    # Second-best BLA gap — shows how dominant the selected example is.
                    "runner_up_gap": bla_runner_up,
                }
                diff_illustrative["bloom_bla_gap"] = ax2_entry

            illustrative_samples[diff] = diff_illustrative

        print(f"\nIllustrative contrast samples selected per difficulty:")
        for diff, axes in illustrative_samples.items():
            for axis_name, entry in axes.items():
                note = entry.get("illustrative_note", {})
                if axis_name == "max_score_gap":
                    print(f"  {diff}/{axis_name}: {entry['file_path']} | gap={note.get('score_gap', '?'):.4f}")
                else:
                    print(f"  {diff}/{axis_name}: {entry['file_path']} | bla_gap={note.get('bloom_bla_gap', '?'):.4f}")

        # ── Blind export: anonymise pipeline labels, strip identity fields ───
        blind_illus: Dict[str, Any] = {}
        illus_key: Dict[str, Any] = {}
        for diff, axes in illustrative_samples.items():
            blind_illus[diff] = {}
            illus_key[diff] = {}
            for axis_name, entry in axes.items():
                _shuffled = list(pipeline_names)
                random.shuffle(_shuffled)
                _label_map = {name: f"pipeline_{chr(65 + i)}" for i, name in enumerate(_shuffled)}
                blind_entry = {k: v for k, v in entry.items() if k not in ("file_path", "generation_model")}
                blind_entry["pipelines"] = dict(sorted(
                    {_label_map[p]: v for p, v in entry["pipelines"].items() if p in _label_map}.items()
                ))
                blind_illus[diff][axis_name] = blind_entry
                illus_key[diff][axis_name] = {
                    "file_path": entry["file_path"],
                    "generation_model": entry["generation_model"],
                    # label -> real pipeline name
                    "pipeline_label_map": {v: k for k, v in _label_map.items()},
                }

        qt_illustrative_path = f"evaluation_results/matched_task_samples_illustrative_contrast_{question_type}.json"
        with open(qt_illustrative_path, "w", encoding="utf-8") as f:
            json.dump(blind_illus, f, indent=4)
        qt_illustrative_key_path = qt_illustrative_path.replace(".json", "_key.json")
        with open(qt_illustrative_key_path, "w", encoding="utf-8") as f:
            json.dump(illus_key, f, indent=4)
        print(f"Illustrative contrast samples (blind) saved to:  {qt_illustrative_path}")
        print(f"Illustrative contrast samples key saved to:      {qt_illustrative_key_path}")

        # ── MATCHED-TASK PAIRWISE EVALUATION (RAW & PRODUCTION MODES) ──────────

        print(f"\n{'='*80}")
        print(f"MATCHED-TASK PAIRWISE EVALUATION — {question_type.upper()}")
        print(f"{'='*80}")
        print("\nFor each (source file, difficulty) task slot, one representative question is")
        print("selected per pipeline and compared directly — fixing the pool-sampling weakness.")
        print("  raw mode        → first generated question (intrinsic generation quality)")
        print("  production mode → highest-scoring question (end-to-end pipeline quality)\n")

        matched_results_all_modes = {}
        for mode in ["raw", "production"]:
            matched_results = run_matched_task_pairwise_evaluation(
                evaluator=evaluator,
                pipeline_names=pipeline_names,
                pairs=pairs,
                task_mapping=task_map,
                discrimination_weights=discrimination_weights,
                question_type=question_type,
                num_bootstrap_samples=NUM_BOOTSTRAP_SAMPLES,
                mode=mode,
            )

            print(f"\n{'='*80}")
            print(f"MATCHED-TASK RESULTS ({mode.upper()} MODE) — {question_type.upper()}")
            print(f"{'='*80}\n")

            print("Pipeline summary (ranked by win rate):")
            sorted_summary = sorted(
                matched_results["summary"].items(),
                key=lambda x: x[1]["win_rate"],
                reverse=True,
            )
            for rank, (pipeline_name, stats) in enumerate(sorted_summary, 1):
                print(f"  {rank}. {pipeline_name}: "
                      f"win={stats['win_rate']:.1%}  "
                      f"loss={stats['loss_rate']:.1%}  "
                      f"tie={stats['tie_rate']:.1%}  "
                      f"(n={stats['total']})")

            print("\nPer-pair detail:")
            for pair_key, result in matched_results["per_pair"].items():
                if result is None:
                    print(f"  {pair_key}: SKIPPED (no matched tasks)")
                    continue
                pa, pb = result["pipeline_a"], result["pipeline_b"]
                print(f"\n  {pair_key} (n={result['total_matched_tasks']} matched tasks):")
                print(f"{pa} wins: {result['wins'][pa]:3d}  ({result['win_rate'][pa]:.1%})")
                print(f"{pb} wins: {result['wins'][pb]:3d}  ({result['win_rate'][pb]:.1%})")
                print(f" Ties (after tie-break): {result['ties']:3d}  ({result['tie_rate']:.1%})")
                print(f" Raw ties (before tie-break): {result['raw_ties']:3d}  ({result['raw_tie_rate']:.1%})")
                print(
                    f" Mean scores:  {pa}={result['mean_score'][pa]:.4f}  "
                    f"{pb}={result['mean_score'][pb]:.4f}"
                    )
                print(f"Mean margin ({pa} - {pb}): {result['mean_margin']:+.4f}")
                ci = result["bootstrap_95ci_win_rate_a"]
                print(f"95% CI for {pa} win rate: [{ci[0]:.1%}, {ci[1]:.1%}]")

                if result["bloom_breakdown"]:
                    print(f"Bloom-level breakdown:")
                    for bl in sorted(result["bloom_breakdown"]):
                        stats = result["bloom_breakdown"][bl]
                        if stats["total"] > 0:
                            print(
                                f"{bl:12s}: {pa}_wins={stats['wins_a']} ({stats['win_rate_a']:.0%}) "
                                f"{pb}_wins={stats['wins_b']} ({stats['win_rate_b']:.0%}) "
                                f"ties={stats['ties']}  n={stats['total']}"
                                )

                if result["difficulty_breakdown"]:
                    print(f"Difficulty breakdown:")
                    for diff in ["easy", "medium", "hard"]:
                        if diff not in result["difficulty_breakdown"]:
                            continue
                        stats = result["difficulty_breakdown"][diff]
                        if stats["total"] > 0:
                            print(
                                f"{diff:6s}: {pa}_wins={stats['wins_a']} ({stats['win_rate_a']:.0%}) "
                                f"{pb}_wins={stats['wins_b']} ({stats['win_rate_b']:.0%}) "
                                f"ties={stats['ties']}  n={stats['total']}"
                                )

            matched_results_all_modes[mode] = matched_results

        # Save matched-task results (per_task_results stripped for file size)
        qt_matched_path = f"evaluation_results/matched_task_results_{question_type}.json"
        serialisable = {}
        for mode, mr in matched_results_all_modes.items():
            serialisable[mode] = {
                "mode": mr["mode"],
                "summary": mr["summary"],
                "per_pair": {
                    pk: {k: v for k, v in pv.items() if k != "per_task_results"}
                    for pk, pv in mr["per_pair"].items()
                    if pv is not None
                },
            }
        with open(qt_matched_path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=4)
        print(f"\nMatched-task results saved to: {qt_matched_path}")

        # Save yield metrics alongside criterion metrics (append to existing file)
        with open(qt_criterion_metrics_path, "r", encoding="utf-8") as f:
            existing_metrics = json.load(f)
        existing_metrics["yield_metrics"] = yield_metrics
        with open(qt_criterion_metrics_path, "w", encoding="utf-8") as f:
            json.dump(existing_metrics, f, indent=4)
        print(f"Yield metrics appended to: {qt_criterion_metrics_path}")

        # ── INTER-RATER RELIABILITY (evaluator model agreement) ───────────────

        print(f"\n{'='*80}")
        print(f"INTER-RATER RELIABILITY — {question_type.upper()}")
        print(f"{'='*80}\n")

        non_bloom_criteria = [
            c for c in criteria
            if c != "bloom_level_alignment"
        ]

        # Agreement over ordinary non-Bloom rubric criteria
        inter_rater_criterion = compute_inter_rater_metrics(
            pipeline_names=pipeline_names,
            all_criterion_results=qt_cr,
            criteria=non_bloom_criteria,
            include_bloom_alignment=False,
        )

        # Agreement over soft Bloom alignment score, i.e. BLA score bucket agreement
        inter_rater_bla = compute_inter_rater_metrics(
            pipeline_names=pipeline_names,
            all_criterion_results=qt_cr,
            criteria=[],
            include_bloom_alignment=True,
        )

        # Agreement over inferred Bloom-level labels
        inter_rater_bloom = compute_bloom_level_agreement(
            pipeline_names=pipeline_names,
            all_criterion_results=qt_cr,
        )

        for pipeline_name in pipeline_names:
            n_judges = inter_rater_criterion[pipeline_name]["n_evaluator_models"]
            print(f"  Pipeline: {pipeline_name}  ({n_judges} evaluator model(s))")
            if n_judges < 2:
                print("    (only one evaluator model — inter-rater metrics not applicable)")
                continue

            pc = inter_rater_criterion[pipeline_name]["per_criterion"]
            for crit in non_bloom_criteria:
                stats = pc.get(crit, {})
                agree = stats.get("agreement_rate")
                std = stats.get("mean_std")
                print(f"    {crit:<40} agree={agree:.4f}  mean_std={std:.4f}  n={stats.get('n', 0)}")

            bla_stats = inter_rater_bla[pipeline_name]["per_criterion"].get("bloom_level_alignment", {})
            print(
                f"    {'bloom_level_alignment_score':<40} "
                f"agree={bla_stats.get('agreement_rate')}  "
                f"mean_std={bla_stats.get('mean_std')}  "
                f"n={bla_stats.get('n', 0)}"
            )

            bloom_agree = inter_rater_bloom[pipeline_name]["agreement_rate"]
            bloom_pairwise = inter_rater_bloom[pipeline_name]["pairwise_percent_agreement"]
            print(f"    {'bloom_level_label_exact':<40} agree={bloom_agree}")
            print(f"    {'bloom_level_label_pairwise':<40} pairwise={bloom_pairwise}")

        # Append inter-rater metrics to the per-type criterion metrics file
        with open(qt_criterion_metrics_path, "r", encoding="utf-8") as f:
            existing_metrics = json.load(f)
        existing_metrics["inter_rater_criterion_metrics"] = inter_rater_criterion
        existing_metrics["inter_rater_bla_score_agreement"] = inter_rater_bla
        existing_metrics["inter_rater_bloom_level_agreement"] = inter_rater_bloom
        with open(qt_criterion_metrics_path, "w", encoding="utf-8") as f:
            json.dump(existing_metrics, f, indent=4)
        print(f"\nInter-rater metrics appended to: {qt_criterion_metrics_path}")

        print(f"\n{'='*80}")
        print(f"EVALUATION COMPLETE — {question_type.upper()}")
        print(f"{'='*80}\n")

    # All question types done
    print("All evaluations completed successfully!")
    print(f"\nResults saved to:")
    for qt in question_types_to_dir.keys():
        print(f"  - evaluation_results/criterion_results_{qt}.json")
        print(f"  - evaluation_results/criterion_results_metrics_{qt}.json  (includes yield metrics)")
        print(f"  - evaluation_results/matched_task_samples_representative_{qt}.json  (blind)")
        print(f"  - evaluation_results/matched_task_samples_representative_{qt}_key.json  (identity map)")
        print(f"  - evaluation_results/matched_task_samples_illustrative_contrast_{qt}.json  (blind)")
        print(f"  - evaluation_results/matched_task_samples_illustrative_contrast_{qt}_key.json  (identity map)")
        print(f"  - evaluation_results/matched_task_results_{qt}.json  (raw & production modes)")