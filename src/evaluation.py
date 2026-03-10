"""
evaluation.py
-------------
Evaluation metrics for recommendation quality and diversity.

Metrics implemented:
  - Precision@K / Recall@K  (accuracy; needs held-out test data)
  - Category Coverage         (diversity; % of categories in recs)
  - Intra-List Diversity      (ILD; average pairwise dissimilarity)
  - Novelty                   (how unexpected the recs are)
  - Bubble Reduction Score    (before vs. after diversity-aware recs)

The main entry point `evaluate_and_compare` runs both standard CF and
diversity-aware CF for a set of users and prints a side-by-side comparison.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

from src.recommender import Recommender
from src.filter_bubble_detector import FilterBubbleDetector, bubble_score, category_entropy, _category_distribution


# ---------------------------------------------------------------------------
# Individual Metrics
# ---------------------------------------------------------------------------

def precision_at_k(recommended: List[int], relevant: List[int]) -> float:
    """
    Fraction of recommended items that are relevant.

    Parameters
    ----------
    recommended : List[int]
        Recommended video_ids.
    relevant : List[int]
        Ground-truth relevant video_ids (held-out test set).

    Returns
    -------
    float in [0, 1].
    """
    if not recommended:
        return 0.0
    hits = len(set(recommended) & set(relevant))
    return hits / len(recommended)


def recall_at_k(recommended: List[int], relevant: List[int]) -> float:
    """
    Fraction of relevant items that appear in the recommendations.

    Returns
    -------
    float in [0, 1].
    """
    if not relevant:
        return 0.0
    hits = len(set(recommended) & set(relevant))
    return hits / len(relevant)


def category_coverage(
    recommended: List[int], video_df: pd.DataFrame, all_categories: List[str]
) -> float:
    """
    Fraction of all categories covered by *recommended* videos.

    Parameters
    ----------
    recommended : List[int]
        Recommended video_ids.
    video_df : pd.DataFrame
    all_categories : List[str]

    Returns
    -------
    float in [0, 1].
    """
    if not recommended:
        return 0.0
    rec_cats = set(video_df.loc[recommended, "category"].values)
    return len(rec_cats) / len(all_categories)


def intra_list_diversity(
    recommended: List[int], video_df: pd.DataFrame
) -> float:
    """
    Intra-List Diversity (ILD): average pairwise category dissimilarity.

    ILD = 1 means every pair of recommended videos is from a different category.
    ILD = 0 means all recommended videos are from the same category.

    Returns
    -------
    float in [0, 1].
    """
    if len(recommended) < 2:
        return 0.0
    cats = video_df.loc[recommended, "category"].values
    n = len(cats)
    dissimilar_pairs = sum(
        1 for i in range(n) for j in range(i + 1, n) if cats[i] != cats[j]
    )
    total_pairs = n * (n - 1) / 2
    return dissimilar_pairs / total_pairs


def novelty_score(
    recommended: List[int],
    interaction_matrix: np.ndarray,
) -> float:
    """
    Novelty: how "unpopular" (rare) the recommended videos are.

    Computed as the average negative log-popularity of each recommended item.
    Popular items get a low novelty score; rare items get a high score.

    Returns
    -------
    float  (higher = more novel).
    """
    if not recommended:
        return 0.0
    num_users = interaction_matrix.shape[0]
    popularity = (interaction_matrix != 0).sum(axis=0) / num_users  # (videos,)
    novelties = []
    for vid in recommended:
        pop = popularity[vid]
        if pop > 0:
            novelties.append(-np.log2(pop))
        else:
            novelties.append(-np.log2(1 / num_users))  # treat as seen by one user
    return float(np.mean(novelties))


# ---------------------------------------------------------------------------
# Bubble Reduction
# ---------------------------------------------------------------------------

def bubble_reduction(
    user_id: int,
    standard_recs: List[int],
    diverse_recs: List[int],
    interaction_matrix: np.ndarray,
    video_df: pd.DataFrame,
    categories: List[str],
) -> Dict:
    """
    Measure the bubble reduction achieved by diversity-aware recommendations.

    Simulates adding the recommended videos to the user's history and
    re-computes the bubble score.

    Returns
    -------
    dict with keys:
        baseline_bubble_score, standard_cf_bubble_score,
        diverse_cf_bubble_score, reduction_vs_standard
    """
    num_categories = len(categories)

    def simulated_bubble(recs: List[int]) -> float:
        """Compute bubble score after appending *recs* to user history."""
        # Clone the user's interaction row and add new interactions
        simulated_row = interaction_matrix[user_id].copy()
        for vid in recs:
            simulated_row[vid] += 1.0  # treat as a watch
        # Temporarily build a one-row matrix
        sim_matrix = np.array([simulated_row])
        dist = _category_distribution(0, sim_matrix, video_df, categories)
        ent = category_entropy(dist)
        return bubble_score(ent, num_categories)

    # Baseline: current bubble score
    dist = _category_distribution(user_id, interaction_matrix, video_df, categories)
    ent = category_entropy(dist)
    baseline = bubble_score(ent, num_categories)

    std_score = simulated_bubble(standard_recs)
    div_score = simulated_bubble(diverse_recs)

    return {
        "baseline_bubble_score": round(baseline, 4),
        "standard_cf_bubble_score": round(std_score, 4),
        "diverse_cf_bubble_score": round(div_score, 4),
        "reduction_vs_standard": round(std_score - div_score, 4),
    }


# ---------------------------------------------------------------------------
# Main Evaluation Routine
# ---------------------------------------------------------------------------

def evaluate_and_compare(
    recommender: Recommender,
    interaction_matrix: np.ndarray,
    video_df: pd.DataFrame,
    categories: List[str],
    test_user_ids: Optional[List[int]] = None,
    num_eval_users: int = 50,
    top_k: int = 10,
    seed: int = 42,
) -> Dict:
    """
    Run both standard CF and diversity-aware CF for *num_eval_users* users
    and compute average metrics for each.

    Note on Precision/Recall: these metrics use the user's *entire* positive
    interaction history as the "relevant" set, not a proper held-out split.
    Because the CF model was trained on this same data, seen items are excluded
    from recommendations, so precision/recall will read close to zero and should
    be interpreted as an approximate lower bound rather than a true accuracy
    measure. The diversity metrics (coverage, ILD, novelty, bubble reduction)
    are the primary indicators of system effectiveness here.

    Returns
    -------
    dict with keys "standard" and "diverse", each containing avg metrics.
    """
    rng = np.random.default_rng(seed)
    num_users = interaction_matrix.shape[0]

    if test_user_ids is None:
        test_user_ids = rng.choice(num_users, size=num_eval_users, replace=False).tolist()

    metrics_std: Dict[str, List[float]] = {
        "precision": [], "recall": [], "coverage": [],
        "ild": [], "novelty": [], "bubble_score": [],
    }
    metrics_div: Dict[str, List[float]] = {
        "precision": [], "recall": [], "coverage": [],
        "ild": [], "novelty": [], "bubble_score": [],
    }
    bubble_reductions: List[float] = []

    for uid in test_user_ids:
        # Build a simple held-out set: videos with positive interactions
        # (in a real system you'd do a proper train/test split)
        liked = np.where(interaction_matrix[uid] > 0)[0].tolist()
        if len(liked) < 5:
            continue  # skip users with too few interactions

        # Standard CF
        result_std = recommender.recommend(uid, apply_diversity=False)
        recs_std = result_std["recommendations"]

        # Diversity-aware CF
        result_div = recommender.recommend(uid, apply_diversity=True)
        recs_div = result_div["recommendations"]

        # Accuracy metrics (note: since we haven't done a real train/test split,
        # this is an approximation — used mainly to show the trade-off)
        metrics_std["precision"].append(precision_at_k(recs_std, liked))
        metrics_std["recall"].append(recall_at_k(recs_std, liked))
        metrics_div["precision"].append(precision_at_k(recs_div, liked))
        metrics_div["recall"].append(recall_at_k(recs_div, liked))

        # Diversity metrics
        metrics_std["coverage"].append(category_coverage(recs_std, video_df, categories))
        metrics_std["ild"].append(intra_list_diversity(recs_std, video_df))
        metrics_std["novelty"].append(novelty_score(recs_std, interaction_matrix))

        metrics_div["coverage"].append(category_coverage(recs_div, video_df, categories))
        metrics_div["ild"].append(intra_list_diversity(recs_div, video_df))
        metrics_div["novelty"].append(novelty_score(recs_div, interaction_matrix))

        # Bubble scores post-recommendation
        br = bubble_reduction(uid, recs_std, recs_div, interaction_matrix, video_df, categories)
        metrics_std["bubble_score"].append(br["standard_cf_bubble_score"])
        metrics_div["bubble_score"].append(br["diverse_cf_bubble_score"])
        bubble_reductions.append(br["reduction_vs_standard"])

    def avg(lst: List[float]) -> float:
        return round(float(np.mean(lst)), 4) if lst else 0.0

    return {
        "standard": {k: avg(v) for k, v in metrics_std.items()},
        "diverse": {k: avg(v) for k, v in metrics_div.items()},
        "avg_bubble_reduction": round(float(np.mean(bubble_reductions)), 4) if bubble_reductions else 0.0,
        "num_users_evaluated": len(test_user_ids),
    }


def print_comparison_table(evaluation_results: Dict) -> None:
    """
    Print a human-readable comparison table from *evaluate_and_compare* output.
    """
    std = evaluation_results["standard"]
    div = evaluation_results["diverse"]
    n = evaluation_results["num_users_evaluated"]

    print("\n" + "=" * 65)
    print(f"  EVALUATION RESULTS  ({n} users)")
    print("=" * 65)
    print(f"{'Metric':<28} {'Standard CF':>14} {'Diverse CF':>14}")
    print("-" * 65)

    metrics = [
        ("Precision@K",          "precision"),
        ("Recall@K",             "recall"),
        ("Category Coverage",    "coverage"),
        ("Intra-List Diversity", "ild"),
        ("Novelty Score",        "novelty"),
        ("Post-Rec Bubble Score","bubble_score"),
    ]

    for label, key in metrics:
        s_val = std.get(key, 0.0)
        d_val = div.get(key, 0.0)
        # Mark improvements with an arrow
        if key == "bubble_score":
            arrow = "↓" if d_val < s_val else ("↑" if d_val > s_val else "=")
        else:
            arrow = "↑" if d_val > s_val else ("↓" if d_val < s_val else "=")
        print(f"  {label:<26} {s_val:>14.4f} {d_val:>13.4f} {arrow}")

    print("-" * 65)
    avg_red = evaluation_results.get("avg_bubble_reduction", 0.0)
    print(f"  {'Avg Bubble Reduction':<26} {avg_red:>14.4f}")
    print("=" * 65)
