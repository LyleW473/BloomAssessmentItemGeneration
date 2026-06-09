import random
from src.evaluation.assessment.saq_evaluation import SAQEvaluator
from src.evaluation.assessment.helper_functions import select_question_for_task
from typing import List, Tuple, Dict, Any

def run_matched_task_pairwise_evaluation(
        evaluator:SAQEvaluator,
        pipeline_names:List[str],
        pairs:List[Tuple[str, str]],
        task_mapping:Dict[Tuple[str, str], Dict[str, Dict[str, Dict[str, Any]]]],
        discrimination_weights:Dict[str, float],
        question_type:str,
        num_bootstrap_samples:int=1000,
        mode:str="raw",
    ) -> Dict[str, Any]:
    """
    Runs matched-task pairwise evaluation over a GLOBAL matched set with bootstrap
    resampling at the triple level, reporting results for every (pipeline, model) combination.

    A global matched set T = {(f, d, g)} is constructed first, keeping only triples
    where ALL pipelines have non-empty outputs for generation model g on task slot (f, d).
    This guarantees a common, consistent denominator for every pipeline and every pair.

    For each triple in T, one representative output per pipeline is selected according to
    'mode' (raw = first candidate; production = highest weighted-score candidate). All
    pairwise comparisons are then performed within each triple, so every pair is evaluated
    on exactly the same N triples.

    Two levels of bootstrap resampling are performed:
    - Overall  - resample T to obtain per-pair 95% CIs (aggregated over models).
    - Per-model - for each model g, resample T_g = {t ∈ T : t.model = g} to obtain
                  95% CIs for every (pipeline, model) and (pair, model) combination.
                  This correctly conditions on g and yields statistically valid intervals
                  for the per-(pipeline, llm) table without conflating model variance.

    Modes:
        raw        = first generated candidate; measures intrinsic generation quality
        production = highest-scoring candidate; measures end-to-end pipeline quality

    Args:
        evaluator (SAQEvaluator): SAQEvaluator instance (used for criteria lists and Bloom constants)
        pipeline_names (List[str]): List of all pipeline names in the experiment
        pairs (List[Tuple[str, str]]): List of (pipeline_a, pipeline_b) pairs to compare
        task_mapping (Dict): Output of build_matched_task_mapping -(file_path, difficulty) -> pipeline -> model -> {"questions", "criteria"}
        discrimination_weights (Dict[str, float]): criterion_name -> normalised weight (sum to 1.0)
        question_type (str): "saqs" or "sbqs"
        num_bootstrap_samples (int): number of bootstrap resamples (applied at both levels)
        mode (str): "raw" or "production"
    Returns:
        Dict[str, Any]: Dict with:
            "mode": the mode used,
            "summary": pipeline_name -> {
                win/loss/tie counts and rates over T (all models),
                "per_model": {
                    model_name -> {
                        win/loss/tie counts and rates conditioned on this model,
                        "bootstrap_95ci": [lo, hi]   ← per-(pipeline, model) CI
                    }
                }
            },
            "per_pair": pair_key -> {
                overall win/loss/tie rates with "bootstrap_95ci_win_rate_a",
                "per_model": {
                    model_name -> {
                        wins_a/b, win_rates, mean_scores,
                        "bootstrap_95ci_a": [lo, hi]   ← per-(pair, model) CI
                    }
                },
                bloom_breakdown, difficulty_breakdown, per_task_results
            }
    """
    if question_type == "saqs":
        TIE_BREAK_PRIORITY = evaluator.TIE_BREAK_PRIORITY_SAQ
    elif question_type == "sbqs":
        TIE_BREAK_PRIORITY = evaluator.TIE_BREAK_PRIORITY_SBQ
    else:
        raise ValueError(f"Unsupported question type: {question_type}")

    EPS = 1e-8

    def _weighted_score(metrics:Dict[str, Any]) -> float:
        """
        Computes the weighted score for a question based on its criterion metrics and the provided discrimination weights.

        Args:
            metrics (Dict[str, Any]): Dictionary of criterion_name -> metric_value for the question.
        Returns:
            float: The weighted score computed as the sum of (metric_value * weight) for each criterion, normalized by the total weight. 
                   If total weight is zero or negative, returns 0.
        """
        total_w = sum(discrimination_weights.values())
        if total_w <= 0:
            return 0.0
        return sum(metrics.get(k, 0.0) * w for k, w in discrimination_weights.items()) / total_w

    def _determine_winner(
            score_a:float,
            score_b:float,
            metrics_a:Dict[str, Any],
            metrics_b:Dict[str, Any]
            ) -> Tuple[str, bool]:
        """
        Determines the winner between two pipelines based on their scores and applies tie-breaking if needed.

        Args:
            score_a (float): The weighted score for pipeline A.
            score_b (float): The weighted score for pipeline B.
            metrics_a (Dict): The original metrics dictionary for pipeline A (used for tie-breaking).
            metrics_b (Dict): The original metrics dictionary for pipeline B (used for tie-breaking).
        Returns:
            Tuple[str, bool]: The winner label ("A", "B", or "Tie") and a boolean indicating if it was a raw tie before tie-breaking.
                              (winner, is_raw_tie)
        """
        is_raw_tie = abs(score_a - score_b) <= EPS
        if score_a > score_b + EPS:
            return "A", is_raw_tie
        if score_b > score_a + EPS:
            return "B", is_raw_tie
        # Scores are equal — apply lexicographic tie-breaking
        for key in TIE_BREAK_PRIORITY:
            va = float(metrics_a.get(key, 0))
            vb = float(metrics_b.get(key, 0))
            if va > vb + EPS:
                return "A", is_raw_tie
            if vb > va + EPS:
                return "B", is_raw_tie
        return "Tie", is_raw_tie

    # Step 1: Build GLOBAL matched set
    # T = {(f, d, g)} where ALL pipelines have non-empty outputs for model g
    global_triples: List[Tuple] = []
    for task_key in sorted(task_mapping.keys()):
        p_to_m_to_data = task_mapping[task_key]
        if not all(p in p_to_m_to_data for p in pipeline_names):
            continue
        model_sets = [set(p_to_m_to_data[p].keys()) for p in pipeline_names]
        common_models = sorted(set.intersection(*model_sets))
        for model_name in common_models:
            if all(
                p_to_m_to_data[p][model_name].get("questions")
                and p_to_m_to_data[p][model_name].get("criteria")
                for p in pipeline_names
            ):
                global_triples.append((task_key, model_name))

    print(f"\n[{mode}] Global matched set: {len(global_triples)} triples (f,d,g) "
          f"across {len(pipeline_names)} pipelines")

    if not global_triples:
        print(f"  [{mode}] No globally matched triples found. Returning empty result.")
        empty_summary = {
            p: {
                "wins": 0, "losses": 0, "ties": 0, "raw_ties": 0, "total": 0,
                "win_rate": 0.0, "loss_rate": 0.0, "tie_rate": 0.0, "per_model": {}
            }
            for p in pipeline_names
        }
        return {
            "mode": mode,
            "summary": empty_summary,
            "per_pair": {f"{pa}_vs_{pb}": None for pa, pb in pairs},
        }

    # Step 2: Score all pipelines per triple + compute all pairwise outcomes ─
    per_triple_data: List[Dict[str, Any]] = []
    for (task_key, model_name) in global_triples:
        p_to_m_to_data = task_mapping[task_key]

        pipeline_info: Dict[str, Any] = {}
        for p in pipeline_names:
            data = p_to_m_to_data[p][model_name]
            q, crit = select_question_for_task(
                data["questions"], data["criteria"], mode, discrimination_weights
            )
            metrics = crit.get("metrics", {})
            pipeline_info[p] = {
                "q": q,
                "crit": crit,
                "metrics": metrics,
                "score": _weighted_score(metrics),
            }

        # Bloom level — prefer judged level from the first pipeline's criterion result
        crit_0 = pipeline_info[pipeline_names[0]]["crit"]
        q_0    = pipeline_info[pipeline_names[0]]["q"]
        bloom = (
            crit_0.get("bloom_alignment_details", {}).get("majority_true_bloom_level")
            or q_0.get("bloom_level")
            or "unknown"
        )
        if bloom in evaluator.BLOOM_LEVELS_SAME:
            bloom = evaluator.BLOOM_LEVELS_SAME[bloom]

        # All pairwise outcomes for this triple
        pair_outcomes: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for (pa, pb) in pairs:
            winner, is_raw_tie = _determine_winner(
                pipeline_info[pa]["score"], pipeline_info[pb]["score"],
                pipeline_info[pa]["metrics"], pipeline_info[pb]["metrics"],
            )
            pair_outcomes[(pa, pb)] = {
                "winner": winner,       # "A" (pa wins), "B" (pb wins), or "Tie"
                "is_raw_tie": is_raw_tie,
                "score_a": pipeline_info[pa]["score"],
                "score_b": pipeline_info[pb]["score"],
            }

        per_triple_data.append({
            "task_key": task_key,
            "model": model_name,
            "bloom_level": bloom,
            "difficulty": task_key[1],
            "pair_outcomes": pair_outcomes,
            "pipeline_scores": {p: pipeline_info[p]["score"] for p in pipeline_names},
        })

    # Step 3: Aggregate pipeline-level and per-pair stats over the global set ─

    # Pipeline summary: each comparison in any pair contributes one win/loss/tie
    pipeline_summary: Dict[str, Dict[str, Any]] = {
        p: {"wins": 0, "losses": 0, "ties": 0, "raw_ties": 0, "total": 0}
        for p in pipeline_names
    }
    pipeline_per_model: Dict[str, Dict[str, Any]] = {p: {} for p in pipeline_names}

    # Per-pair accumulators (lists kept for mean score / bootstrap)
    pair_accum: Dict[Tuple[str, str], Dict[str, Any]] = {
        (pa, pb): {
            "wins_a": 0, "wins_b": 0, "ties": 0, "raw_ties": 0, "total": 0,
            "scores_a": [], "scores_b": [], "margins": [], "per_model": {},
        }
        for (pa, pb) in pairs
    }
    pair_bloom: Dict[Tuple[str, str], Dict[str, Any]] = {pair: {} for pair in pairs}
    pair_diff:  Dict[Tuple[str, str], Dict[str, Any]] = {pair: {} for pair in pairs}

    for triple in per_triple_data:
        model = triple["model"]
        bloom = triple["bloom_level"]
        diff  = triple["difficulty"]

        for (pa, pb) in pairs:
            outcome  = triple["pair_outcomes"][(pa, pb)]
            winner   = outcome["winner"]
            raw_tie  = outcome["is_raw_tie"]
            score_a  = outcome["score_a"]
            score_b  = outcome["score_b"]
            acc      = pair_accum[(pa, pb)]

            # pair accumulator
            acc["total"]    += 1
            acc["raw_ties"] += int(raw_tie)
            acc["scores_a"].append(score_a)
            acc["scores_b"].append(score_b)
            acc["margins"].append(score_a - score_b)
            if   winner == "A": acc["wins_a"] += 1
            elif winner == "B": acc["wins_b"] += 1
            else:               acc["ties"]   += 1

            # pair per-model
            if model not in acc["per_model"]:
                acc["per_model"][model] = {
                    "wins_a": 0, "wins_b": 0, "ties": 0, "raw_ties": 0, "total": 0,
                    "_scores_a": [], "_scores_b": [],
                }
            pm = acc["per_model"][model]
            pm["total"]    += 1
            pm["raw_ties"] += int(raw_tie)
            pm["_scores_a"].append(score_a)
            pm["_scores_b"].append(score_b)
            if   winner == "A": pm["wins_a"] += 1
            elif winner == "B": pm["wins_b"] += 1
            else:               pm["ties"]   += 1

            # bloom breakdown
            if bloom not in pair_bloom[(pa, pb)]:
                pair_bloom[(pa, pb)][bloom] = {
                    "wins_a": 0, "wins_b": 0, "ties": 0, "raw_ties": 0, "total": 0
                }
            bb = pair_bloom[(pa, pb)][bloom]
            bb["total"]    += 1
            bb["raw_ties"] += int(raw_tie)
            if   winner == "A": bb["wins_a"] += 1
            elif winner == "B": bb["wins_b"] += 1
            else:               bb["ties"]   += 1

            # difficulty breakdown
            if diff not in pair_diff[(pa, pb)]:
                pair_diff[(pa, pb)][diff] = {"wins_a": 0, "wins_b": 0, "ties": 0, "total": 0}
            db = pair_diff[(pa, pb)][diff]
            db["total"] += 1
            if   winner == "A": db["wins_a"] += 1
            elif winner == "B": db["wins_b"] += 1
            else:               db["ties"]   += 1

            # pipeline-level summary (same global denominator for every pipeline)
            for p, is_pa in [(pa, True), (pb, False)]:
                pipeline_summary[p]["total"]    += 1
                pipeline_summary[p]["raw_ties"] += int(raw_tie)
                if winner == "Tie":
                    pipeline_summary[p]["ties"] += 1
                elif (winner == "A") == is_pa:
                    pipeline_summary[p]["wins"] += 1
                else:
                    pipeline_summary[p]["losses"] += 1

                if model not in pipeline_per_model[p]:
                    pipeline_per_model[p][model] = {
                        "wins": 0, "losses": 0, "ties": 0, "raw_ties": 0, "total": 0
                    }
                pm_s = pipeline_per_model[p][model]
                pm_s["total"]    += 1
                pm_s["raw_ties"] += int(raw_tie)
                if winner == "Tie":
                    pm_s["ties"] += 1
                elif (winner == "A") == is_pa:
                    pm_s["wins"] += 1
                else:
                    pm_s["losses"] += 1

    # Compute rates for pipeline summary
    for p, stats in pipeline_summary.items():
        t = stats["total"]
        stats["win_rate"]  = stats["wins"]   / t if t > 0 else 0.0
        stats["loss_rate"] = stats["losses"] / t if t > 0 else 0.0
        stats["tie_rate"]  = stats["ties"]   / t if t > 0 else 0.0
    for p in pipeline_names:
        for m, stats in pipeline_per_model[p].items():
            t = stats["total"]
            stats["win_rate"]  = stats["wins"]   / t if t > 0 else 0.0
            stats["loss_rate"] = stats["losses"] / t if t > 0 else 0.0
            stats["tie_rate"]  = stats["ties"]   / t if t > 0 else 0.0
        pipeline_summary[p]["per_model"] = pipeline_per_model[p]

    # Finalise per-model and breakdown rates in pair accumulators
    for (pa, pb) in pairs:
        acc = pair_accum[(pa, pb)]
        for model, pm in acc["per_model"].items():
            t = pm["total"]
            pm["win_rate_a"]   = pm["wins_a"] / t if t > 0 else 0.0
            pm["win_rate_b"]   = pm["wins_b"] / t if t > 0 else 0.0
            pm["mean_score_a"] = sum(pm.pop("_scores_a")) / t if t > 0 else 0.0
            pm["mean_score_b"] = sum(pm.pop("_scores_b")) / t if t > 0 else 0.0
        for bl, stats in pair_bloom[(pa, pb)].items():
            t = stats["total"]
            stats["win_rate_a"] = stats["wins_a"] / t if t > 0 else 0.0
            stats["win_rate_b"] = stats["wins_b"] / t if t > 0 else 0.0
        for d, stats in pair_diff[(pa, pb)].items():
            t = stats["total"]
            stats["win_rate_a"] = stats["wins_a"] / t if t > 0 else 0.0
            stats["win_rate_b"] = stats["wins_b"] / t if t > 0 else 0.0

    # Step 4: Bootstrap
    #
    # 4a. Overall bootstrap — resample all triples T to get per-pair CIs
    #     (aggregated over all models).
    #
    # 4b. Per-model bootstrap — for each model g, resample T_g = {t ∈ T : t.model = g}
    #     independently.  This produces:
    #       • per-(pair, model) CIs  → per_pair[pair_key]["per_model"][g]["bootstrap_95ci_a"]
    #       • per-(pipeline, model) CIs → summary[p]["per_model"][g]["bootstrap_95ci"]
    #     allowing statistically valid confidence intervals for every (pipeline, llm)
    #     combination reported in the results table.

    def _ci95(values: List[float]) -> List[float]:
        s = sorted(values)
        n = len(s)
        return [s[int(0.025 * n)], s[int(0.975 * n)]]

    # 4a: Overall bootstrap (over all triples T)
    N = len(per_triple_data)
    rng = random.Random(42)
    pair_boot: Dict[Tuple[str, str], List[float]] = {pair: [] for pair in pairs}

    for _ in range(num_bootstrap_samples):
        boot = rng.choices(per_triple_data, k=N)
        for (pa, pb) in pairs:
            wins_a = sum(1 for t in boot if t["pair_outcomes"][(pa, pb)]["winner"] == "A")
            pair_boot[(pa, pb)].append(wins_a / N)

    per_pair_ci: Dict[Tuple[str, str], List[float]] = {
        pair: _ci95(pair_boot[pair]) for pair in pairs
    }

    # 4b: Per-model bootstrap (fix g, resample T_g)
    models_all: List[str] = sorted(set(t["model"] for t in per_triple_data))
    per_model_triple_data: Dict[str, List[Dict[str, Any]]] = {
        g: [t for t in per_triple_data if t["model"] == g] for g in models_all
    }

    # Bootstrapped win-rate distributions
    pair_model_boot: Dict[str, Dict[Tuple[str, str], List[float]]] = {
        g: {pair: [] for pair in pairs} for g in models_all
    }
    n_opponents = len(pipeline_names) - 1
    pipeline_model_boot: Dict[str, Dict[str, List[float]]] = {
        g: {p: [] for p in pipeline_names} for g in models_all
    }

    for g in models_all:
        T_g = per_model_triple_data[g]
        N_g = len(T_g)
        if N_g == 0:
            continue
        rng_g = random.Random(42)
        for _ in range(num_bootstrap_samples):
            boot_g = rng_g.choices(T_g, k=N_g)

            # Per-pair per-model: count wins_a and wins_b for each pair
            boot_wins_a: Dict[Tuple[str, str], int] = {}
            boot_wins_b: Dict[Tuple[str, str], int] = {}
            for (pa, pb) in pairs:
                wa = sum(1 for t in boot_g if t["pair_outcomes"][(pa, pb)]["winner"] == "A")
                wb = sum(1 for t in boot_g if t["pair_outcomes"][(pa, pb)]["winner"] == "B")
                boot_wins_a[(pa, pb)] = wa
                boot_wins_b[(pa, pb)] = wb
                pair_model_boot[g][(pa, pb)].append(wa / N_g)

            # Per-(pipeline, model): marginal win rate of p conditioned on model g
            # = (wins of p against all opponents in T_g) / (N_g * n_opponents)
            total_g = N_g * n_opponents
            for p in pipeline_names:
                wins_p = sum(
                    boot_wins_a[(pa2, pb2)] if pa2 == p else boot_wins_b[(pa2, pb2)]
                    for (pa2, pb2) in pairs if pa2 == p or pb2 == p
                )
                pipeline_model_boot[g][p].append(wins_p / total_g if total_g > 0 else 0.0)

    # Compute CIs from bootstrap distributions
    per_pair_model_ci: Dict[str, Dict[Tuple[str, str], List[float]]] = {
        g: {pair: _ci95(pair_model_boot[g][pair]) for pair in pairs}
        for g in models_all
    }
    pipeline_model_ci: Dict[str, Dict[str, List[float]]] = {
        g: {p: _ci95(pipeline_model_boot[g][p]) for p in pipeline_names}
        for g in models_all
    }

    # Attach per-model CIs to pipeline_per_model (already stored in pipeline_summary)
    for p in pipeline_names:
        for m, stats in pipeline_per_model[p].items():
            stats["bootstrap_95ci"] = pipeline_model_ci.get(m, {}).get(p, [None, None])
    # (pipeline_summary[p]["per_model"] already points to pipeline_per_model[p])

    # Step 5: Build per_pair results
    per_pair_results: Dict[str, Any] = {}
    for (pa, pb) in pairs:
        pair_key = f"{pa}_vs_{pb}"
        acc = pair_accum[(pa, pb)]
        n   = acc["total"]

        # Attach per-model CIs to the pair's per_model breakdown
        for m, pm in acc["per_model"].items():
            pm["bootstrap_95ci_a"] = per_pair_model_ci.get(m, {}).get((pa, pb), [None, None])
            pm["bootstrap_95ci_b"] = per_pair_model_ci.get(m, {}).get((pa, pb), [None, None])
            # Note: ci_b mirrors ci_a from the other side (1 - win_rate_a is derived, not separately bootstrapped)

        per_task_list = [
            {
                "task_key":   list(t["task_key"]),
                "model":      t["model"],
                "bloom_level": t["bloom_level"],
                "difficulty": t["difficulty"],
                "winner":     t["pair_outcomes"][(pa, pb)]["winner"],
                "is_raw_tie": t["pair_outcomes"][(pa, pb)]["is_raw_tie"],
                "score_a":    t["pair_outcomes"][(pa, pb)]["score_a"],
                "score_b":    t["pair_outcomes"][(pa, pb)]["score_b"],
                "margin":     t["pair_outcomes"][(pa, pb)]["score_a"] - t["pair_outcomes"][(pa, pb)]["score_b"],
            }
            for t in per_triple_data
        ]

        per_pair_results[pair_key] = {
            "pipeline_a": pa,
            "pipeline_b": pb,
            "mode": mode,
            "total_matched_tasks": n,
            "wins": {pa: acc["wins_a"], pb: acc["wins_b"]},
            "ties": acc["ties"],
            "raw_ties": acc["raw_ties"],
            "win_rate": {
                pa: acc["wins_a"] / n if n > 0 else 0.0,
                pb: acc["wins_b"] / n if n > 0 else 0.0,
            },
            "tie_rate":     acc["ties"]     / n if n > 0 else 0.0,
            "raw_tie_rate": acc["raw_ties"] / n if n > 0 else 0.0,
            "mean_score": {
                pa: sum(acc["scores_a"]) / n if n > 0 else 0.0,
                pb: sum(acc["scores_b"]) / n if n > 0 else 0.0,
            },
            "mean_margin":               sum(acc["margins"]) / n if n > 0 else 0.0,
            "bootstrap_95ci_win_rate_a": per_pair_ci[(pa, pb)],
            "bloom_breakdown":           pair_bloom[(pa, pb)],
            "difficulty_breakdown":      pair_diff[(pa, pb)],
            "per_model":                 acc["per_model"],
            "per_task_results":          per_task_list,
        }

    return {
        "mode": mode,
        "summary": pipeline_summary,
        "per_pair": per_pair_results,
    }