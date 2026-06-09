import os
import json
import random
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any

class SAQEvaluator:
    def __init__(self):

        # Load keys
        load_dotenv()
        OPENAI_LLM_API_KEY = os.getenv("OPENAI_LLM_API_KEY", None)
        OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)

        for key in [OPENAI_LLM_API_KEY, OPENAI_BASE_URL]:
            if key is None:
                raise ValueError(f"{key} not found in environment variables")
        
        # Initialise OpenAI client
        self.client = OpenAI(
            api_key=OPENAI_LLM_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        self.TIE_BREAK_PRIORITY_SAQ = [
            "objective_gradability_triple", # Can it be graded consistently with THIS mark scheme?
            "objective_gradability_stem",   # Can it be graded consistently based on the question stem alone (without mark scheme)?
            "mark_scheme_quality",          # Does the scheme actually support good grading?
            "answer_fidelity",              # Does expected answer match the scheme (sanity check)
            "single_task_integrity",        # Avoids double-barrelled questions
            "clarity",                      # Clear stem reduces marking variance
            "course_grounding",             # Anchored to source material
            "bloom_level_alignment",        # Declared vs inferred; important but least 'fatal
        ]
        self.TIE_BREAK_PRIORITY_SBQ = [
            # 1. HARD VALIDITY (must be correct or question is fundamentally broken)
            "objective_gradability_triple",     # Still the most critical (can it be marked consistently?)
            "mark_scheme_quality",              # Scheme must support grading properly
            "answer_fidelity",                  # Answer must align with scheme

            # 2. SCENARIO VALIDITY (SBQ-specific)
            "scenario_relevance_necessity",     # Is scenario REQUIRED to answer?
            "scenario_grounding",               # Is scenario accurate / realistic / course-grounded?

            # 3. QUESTION STRUCTURE
            "objective_gradability_stem",       # Can it be graded from stem alone?
            "single_task_integrity",            # Avoid multi-task ambiguity

            # 4. QUALITY / PRESENTATION
            "clarity",                          # Important but secondary
            "course_grounding",                 # General grounding (less critical than scenario grounding)

            # 5. TARGETING (important but least fatal)
            "bloom_level_alignment"
        ]


        self.BLOOM_LEVELS = ["knowledge", "understanding", "application", "analyze", "synthesis", "evaluation"]
        self.BLOOM_LEVELS_SAME = {
            "create": "synthesis",
            "evaluate": "evaluation"
        }
        
    def run_pairwise_evaluation(
            self, 
            packed_data_list_1:List[Dict[str, Any]], 
            packed_data_list_2:List[Dict[str, Any]],
            criterion_results_1:List[Dict[str, Any]],
            criterion_results_2:List[Dict[str, Any]],
            pipeline_names:List[str],
            K:int,
            question_type:str,
            num_bootstrap_samples:int=1000,
            base_seed:int=42,
            stratify_by_bloom_level:bool=False,
            use_discrimination_weights:bool=True,
            discrimination_weights:Dict[str, float]=None
        ) -> Dict[str, Any]:

        if question_type == "saqs":
            TIE_BREAK_PRIORITY = self.TIE_BREAK_PRIORITY_SAQ
        elif question_type == "sbqs":
            TIE_BREAK_PRIORITY = self.TIE_BREAK_PRIORITY_SBQ
        else:
            raise ValueError(f"Unsupported question type: {question_type}")
        
        def get_bloom_level(criterion_item: Dict) -> str:
            """Extract bloom level, preferring judged/verified level over declared."""
            true_bloom_level = criterion_item["bloom_alignment_details"]["majority_true_bloom_level"]
            if true_bloom_level in self.BLOOM_LEVELS_SAME:
                true_bloom_level = self.BLOOM_LEVELS_SAME[true_bloom_level] # Map "create" to "synthesis" and "evaluate" to "evaluation" for consistency
            assert true_bloom_level in self.BLOOM_LEVELS, f"Unexpected true bloom level: {true_bloom_level}"
            return true_bloom_level
        
        # Build strata if stratifying
        if stratify_by_bloom_level:
            # print("Building Bloom level strata for stratified sampling...")

            A_strata = {lvl: [] for lvl in self.BLOOM_LEVELS}
            B_strata = {lvl: [] for lvl in self.BLOOM_LEVELS}
            
            for idx, (item, c) in enumerate(zip(packed_data_list_1, criterion_results_1)):
                lvl = get_bloom_level(c)
                if lvl in A_strata:
                    A_strata[lvl].append(idx)
            
            for idx, (item, c) in enumerate(zip(packed_data_list_2, criterion_results_2)):
                lvl = get_bloom_level(c)
                if lvl in B_strata:
                    B_strata[lvl].append(idx)
            
            # Only use levels present in BOTH pipelines
            eligible_levels = [lvl for lvl in self.BLOOM_LEVELS if A_strata[lvl] and B_strata[lvl]]
            eligible_levels = sorted(eligible_levels, key=lambda lvl: self.BLOOM_LEVELS.index(lvl)) # Sort by Bloom's hierarchy
            
            if not eligible_levels:
                raise ValueError("No common Bloom levels found in both pipelines for stratified sampling")
            
            # print(f"Eligible levels for stratified sampling: {eligible_levels}")
            # for lvl in eligible_levels:
            #     print(f"  {lvl}: A={len(A_strata[lvl])}, B={len(B_strata[lvl])}")
        
        all_bootstrap_results = []
        
        # Perform multiple bootstrap samples
        for bootstrap_iter in range(num_bootstrap_samples):
            seed = base_seed + bootstrap_iter
            sampling_rng = random.Random(seed)
            positional_rng = random.Random(seed + 100)
            
            # print(f"\n=== Bootstrap iteration {bootstrap_iter + 1}/{num_bootstrap_samples} (seed={seed}) ===")
            
            # Sample K indices based on stratification strategy
            if stratify_by_bloom_level:
                # Balanced stratified sampling: split K across eligible levels
                k_per_level = K // len(eligible_levels)
                remainder = K % len(eligible_levels)
                
                sampled_indices_a = []
                sampled_indices_b = []
                
                for j, lvl in enumerate(eligible_levels):
                    # Distribute remainder across first few levels
                    k_lvl = k_per_level + (1 if j < remainder else 0)
                    
                    # Sample with replacement from each stratum
                    sampled_indices_a += sampling_rng.choices(A_strata[lvl], k=k_lvl)
                    sampled_indices_b += sampling_rng.choices(B_strata[lvl], k=k_lvl)

                # Temporary: debug, print sampled levels to verify stratification
                sampled_levels_a = [get_bloom_level(criterion_results_1[i]) for i in sampled_indices_a]
                sampled_levels_b = [get_bloom_level(criterion_results_2[i]) for i in sampled_indices_b]

                from collections import Counter
                ca = Counter(sampled_levels_a)
                cb = Counter(sampled_levels_b)

                # print("Sampled level counts A:", dict(ca))
                # print("Sampled level counts B:", dict(cb))
                # print()
                
                # print(f"Stratified sampling: {k_per_level} per level, {remainder} remainder distributed")

                # Check 2: Verify that the sampled distributions match the intended stratification
                from collections import Counter
                ca = Counter([get_bloom_level(criterion_results_1[i]) for i in sampled_indices_a])
                cb = Counter([get_bloom_level(criterion_results_2[i]) for i in sampled_indices_b])
                assert ca == cb, f"Stratified distributions differ: A={ca}, B={cb}"
                
            else:
                # Unstratified: sample from full range with replacement
                sampled_indices_a = sampling_rng.choices(range(len(packed_data_list_1)), k=K)
                sampled_indices_b = sampling_rng.choices(range(len(packed_data_list_2)), k=K)


            a_items = [packed_data_list_1[i] for i in sampled_indices_a]
            b_items = [packed_data_list_2[i] for i in sampled_indices_b]
            criterion_items_1 = [criterion_results_1[i] for i in sampled_indices_a]
            criterion_items_2 = [criterion_results_2[i] for i in sampled_indices_b]
            
            results = []
            for i in range(K):
                a = {
                    "question": a_items[i]["question"],
                    "bloom_level": criterion_items_1[i]["bloom_alignment_details"]["declared_bloom_level"],
                    "mark_scheme": a_items[i]["mark_scheme"],
                    "expected_answer": a_items[i]["expected_answer"],
                }
                b = {
                    "question": b_items[i]["question"],
                    "bloom_level": criterion_items_2[i]["bloom_alignment_details"]["declared_bloom_level"],
                    "mark_scheme": b_items[i]["mark_scheme"],
                    "expected_answer": b_items[i]["expected_answer"],
                }

                # Randomize presentation order to reduce positional bias
                if positional_rng.random() < 0.5:
                    mapping = {"A": pipeline_names[0], "B": pipeline_names[1]}
                    criterion_results_mapping = {
                        "A": criterion_items_1[i],
                        "B": criterion_items_2[i]
                    }
                else:
                    mapping = {"A": pipeline_names[1], "B": pipeline_names[0]}
                    criterion_results_mapping = {
                        "A": criterion_items_2[i],
                        "B": criterion_items_1[i]
                    }

                # Get criterion results
                results_per_presentation = {}
                for presentation_key in ["A", "B"]:
                    criterion_results = criterion_results_mapping[presentation_key]
                    results_per_presentation[presentation_key] = criterion_results

                # Decide the winner
                # Use discrimination weights if provided, otherwise use RAW_WEIGHTS
                if use_discrimination_weights and discrimination_weights is not None:
                    # Discrimination weights are already normalized to sum to 1.0
                    SCORE_WEIGHTS = discrimination_weights.copy()
                else:
                    # Default equal-weighted approach
                    SCORE_WEIGHTS = {criterion: 1.0 for criterion in TIE_BREAK_PRIORITY}

                def normalise_weights(weights: dict) -> dict:
                    total = sum(weights.values())
                    if total <= 0:
                        raise ValueError(f"Invalid weights sum={total}: {weights}")
                    return {k: v / total for k, v in weights.items()}

                def weighted_score(metrics: dict, weights: dict) -> float:
                    norm_weights = normalise_weights(weights)
                    return sum(norm_weights[k] * float(metrics.get(k, 0)) for k in norm_weights)

                a_metrics = results_per_presentation["A"]["metrics"]
                b_metrics = results_per_presentation["B"]["metrics"]

                score_a = weighted_score(a_metrics, SCORE_WEIGHTS)
                score_b = weighted_score(b_metrics, SCORE_WEIGHTS)

                EPS = 1e-8
                if score_a > score_b + EPS:
                    winner = "A"
                elif score_b > score_a + EPS:
                    winner = "B"
                else:
                    winner = "Tie"
                    for key in TIE_BREAK_PRIORITY:
                        a_val = a_metrics.get(key, 0)
                        b_val = b_metrics.get(key, 0)
                        if a_val > b_val:
                            winner = "A"
                            break
                        elif b_val > a_val:
                            winner = "B"
                            break
                
                if stratify_by_bloom_level:
                    pair_level_a = get_bloom_level(criterion_items_1[i])
                    pair_level_b = get_bloom_level(criterion_items_2[i])
                    pair_level = pair_level_a # In stratified sampling, both should be the same
                    assert pair_level_a == pair_level_b, f"Stratified sampling inconsistency: {pair_level_a} vs {pair_level_b}"
                else:
                    if winner == "Tie": # Tie
                        pair_level = "Tie"
                    else: # Winner's bloom level
                        winner_criterion_results = criterion_results_mapping[winner]
                        pair_level = get_bloom_level(winner_criterion_results)
                
                evaluation_result = {
                    "pair_index": i,
                    "winner": winner,
                    "winner_pipeline": mapping[winner] if winner in mapping else "Tie",
                    "true_bloom_level": pair_level,
                    "score_a": score_a,
                    "score_b": score_b,
                    "scores_per_criterion": results_per_presentation[winner]["metrics"] if winner in ["A", "B"] else {"A": a_metrics, "B": b_metrics},
                }
                results.append(evaluation_result)

            # Calculate metrics for this bootstrap sample
            pipeline_a_wins = sum(1 for r in results if r["winner_pipeline"] == pipeline_names[0])
            pipeline_b_wins = sum(1 for r in results if r["winner_pipeline"] == pipeline_names[1])
            num_ties = sum(1 for r in results if r["winner"] == "Tie")
            pipeline_a_losses = K - pipeline_a_wins - num_ties
            pipeline_b_losses = K - pipeline_b_wins - num_ties
            
            # Calculate mean weighted scores and margins
            total_score_a = sum(r["score_a"] for r in results)
            total_score_b = sum(r["score_b"] for r in results)
            mean_score_a = total_score_a / len(results) if results else 0
            mean_score_b = total_score_b / len(results) if results else 0
            margins = [r["score_a"] - r["score_b"] for r in results]
            mean_margin = sum(margins) / len(margins) if margins else 0
            
            # Per-level breakdown if stratifying
            level_breakdown = {}
            if stratify_by_bloom_level:
                for lvl in eligible_levels:
                    level_results = [r for r in results if r["true_bloom_level"] == lvl]

                    wins_a_lvl = sum(1 for r in level_results if r["winner_pipeline"] == pipeline_names[0])
                    wins_b_lvl = sum(1 for r in level_results if r["winner_pipeline"] == pipeline_names[1])
                    ties_lvl   = sum(1 for r in level_results if r["winner"] == "Tie")
                    losses_a_lvl = len(level_results) - wins_a_lvl - ties_lvl
                    losses_b_lvl = len(level_results) - wins_b_lvl - ties_lvl
                    
                    # Calculate mean scores and margin per level
                    level_score_a = sum(r["score_a"] for r in level_results)
                    level_score_b = sum(r["score_b"] for r in level_results)
                    level_mean_score_a = level_score_a / len(level_results) if level_results else 0
                    level_mean_score_b = level_score_b / len(level_results) if level_results else 0
                    level_margins = [r["score_a"] - r["score_b"] for r in level_results]
                    level_mean_margin = sum(level_margins) / len(level_margins) if level_margins else 0

                    level_breakdown[lvl] = {
                        "wins_a": wins_a_lvl,
                        "wins_b": wins_b_lvl,
                        "losses_a": losses_a_lvl,
                        "losses_b": losses_b_lvl,
                        "ties": ties_lvl,
                        "total": len(level_results),
                        "mean_score_a": level_mean_score_a,
                        "mean_score_b": level_mean_score_b,
                        "mean_margin": level_mean_margin
                    }

            bootstrap_metrics = {
                "bootstrap_iter": bootstrap_iter,
                "seed": seed,
                "stratified": stratify_by_bloom_level,
                "wins": {
                    pipeline_names[0]: pipeline_a_wins,
                    pipeline_names[1]: pipeline_b_wins
                },
                "losses": {
                    pipeline_names[0]: pipeline_a_losses,
                    pipeline_names[1]: pipeline_b_losses
                },
                "num_ties": num_ties,
                "total_comparisons": len(results),
                "mean_score_a": mean_score_a,
                "mean_score_b": mean_score_b,
                "mean_margin": mean_margin,
                "level_breakdown": level_breakdown if stratify_by_bloom_level else None
            }
            
            all_bootstrap_results.append(bootstrap_metrics)
            # print(f"Bootstrap {bootstrap_iter + 1} Metrics: {json.dumps(bootstrap_metrics, indent=4)}")

        # Aggregate results
        total_wins_a = sum(br["wins"][pipeline_names[0]] for br in all_bootstrap_results)
        total_wins_b = sum(br["wins"][pipeline_names[1]] for br in all_bootstrap_results)
        total_losses_a = sum(br["losses"][pipeline_names[0]] for br in all_bootstrap_results)
        total_losses_b = sum(br["losses"][pipeline_names[1]] for br in all_bootstrap_results)
        total_ties = sum(br["num_ties"] for br in all_bootstrap_results)
        total_comparisons = sum(br["total_comparisons"] for br in all_bootstrap_results)
        
        # Aggregate mean scores and margins
        all_mean_scores_a = [br["mean_score_a"] for br in all_bootstrap_results]
        all_mean_scores_b = [br["mean_score_b"] for br in all_bootstrap_results]
        all_mean_margins = [br["mean_margin"] for br in all_bootstrap_results]
        
        overall_mean_score_a = sum(all_mean_scores_a) / len(all_mean_scores_a) if all_mean_scores_a else 0
        overall_mean_score_b = sum(all_mean_scores_b) / len(all_mean_scores_b) if all_mean_scores_b else 0
        overall_mean_margin = sum(all_mean_margins) / len(all_mean_margins) if all_mean_margins else 0
        
        # Aggregate per-level results if stratified
        aggregated_level_breakdown = None
        if stratify_by_bloom_level:
            aggregated_level_breakdown = {}
            for lvl in eligible_levels:
                aggregated_level_breakdown[lvl] = {
                    "wins_a": sum(br["level_breakdown"][lvl]["wins_a"] for br in all_bootstrap_results),
                    "wins_b": sum(br["level_breakdown"][lvl]["wins_b"] for br in all_bootstrap_results),
                    "ties": sum(br["level_breakdown"][lvl]["ties"] for br in all_bootstrap_results),
                    "total": sum(br["level_breakdown"][lvl]["total"] for br in all_bootstrap_results),
                    "mean_score_a": sum(br["level_breakdown"][lvl]["mean_score_a"] for br in all_bootstrap_results) / num_bootstrap_samples,
                    "mean_score_b": sum(br["level_breakdown"][lvl]["mean_score_b"] for br in all_bootstrap_results) / num_bootstrap_samples,
                    "mean_margin": sum(br["level_breakdown"][lvl]["mean_margin"] for br in all_bootstrap_results) / num_bootstrap_samples
                }

        aggregated_metrics = {
            "num_bootstrap_samples": num_bootstrap_samples,
            "stratified": stratify_by_bloom_level,
            "stratify_by": "bloom_level" if stratify_by_bloom_level else None,
            "wins": {
                pipeline_names[0]: total_wins_a,
                pipeline_names[1]: total_wins_b
            },
            "losses": {
                pipeline_names[0]: total_losses_a,
                pipeline_names[1]: total_losses_b
            },
            "num_ties": total_ties,
            "total_comparisons": total_comparisons,
            "win_rate": {
                pipeline_names[0]: total_wins_a / total_comparisons if total_comparisons > 0 else 0,
                pipeline_names[1]: total_wins_b / total_comparisons if total_comparisons > 0 else 0
            },
            "mean_score_a": overall_mean_score_a,
            "mean_score_b": overall_mean_score_b,
            "mean_margin": overall_mean_margin,
            "level_breakdown": aggregated_level_breakdown
        }
        
        # print(f"\n=== Aggregated Pairwise Evaluation Metrics ===")
        # print(f"Pipeline A: {pipeline_names[0]} | Pipeline B: {pipeline_names[1]}")
        # print(f"Stratification by Bloom level: {'Yes' if stratify_by_bloom_level else 'No'}")
        # print(json.dumps(aggregated_metrics, indent=4))
        
        return aggregated_metrics