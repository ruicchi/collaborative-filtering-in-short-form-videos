"""
filter_bubble_detector.py
-------------------------
Detects and quantifies filter bubbles in a user's interaction history.

A "filter bubble" occurs when a user is primarily exposed to content from a
narrow set of categories, reducing information diversity.

Metrics implemented:
1. **Category Entropy** — Shannon entropy of the category distribution.
   - Low entropy  → concentrated in few categories → strong bubble.
   - High entropy → spread across many categories → diverse.
2. **Bubble Score** — Normalised 0-1 score; 1 = fully in a bubble, 0 = perfectly diverse.
3. **Category Concentration Ratio** — Fraction of interactions in the top-1 / top-2 categories.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def _category_distribution(
    user_id: int,
    interaction_matrix: np.ndarray,
    video_df: pd.DataFrame,
    categories: List[str],
) -> np.ndarray:
    """
    Return a probability distribution over categories for *user_id*.

    Only interactions where the weight > 0 (watches / likes) are counted;
    skips (negative weights) are excluded to reflect content the user *chose*
    to engage with.
    """
    user_interactions = interaction_matrix[user_id]
    engaged = np.where(user_interactions > 0)[0]  # video indices with positive signal

    if len(engaged) == 0:
        # No positive interactions: uniform distribution (max diversity)
        return np.ones(len(categories)) / len(categories)

    # Accumulate total interaction weight per category
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    weights = np.zeros(len(categories))
    for vid_idx in engaged:
        cat = video_df.loc[vid_idx, "category"]
        weights[cat_to_idx[cat]] += user_interactions[vid_idx]

    total = weights.sum()
    if total == 0:
        return np.ones(len(categories)) / len(categories)
    return weights / total


def category_entropy(distribution: np.ndarray) -> float:
    """
    Compute Shannon entropy of *distribution* (in bits, base-2).

    Parameters
    ----------
    distribution : np.ndarray
        Probability distribution over categories (must sum to 1).

    Returns
    -------
    float  — entropy value. Max entropy = log2(num_categories).
    """
    # Only include non-zero probabilities to avoid log(0)
    nonzero = distribution[distribution > 0]
    return -float(np.sum(nonzero * np.log2(nonzero)))


def bubble_score(entropy: float, num_categories: int) -> float:
    """
    Normalise entropy to a 0-1 bubble score.

    score = 1 - (entropy / max_entropy)

    A score of 1 means the user only consumes one category (maximum bubble).
    A score of 0 means perfectly uniform category distribution (no bubble).

    Parameters
    ----------
    entropy : float
        Shannon entropy of the user's category distribution.
    num_categories : int
        Total number of categories (used to compute max possible entropy).

    Returns
    -------
    float  in [0, 1].
    """
    max_entropy = np.log2(num_categories)
    if max_entropy == 0:
        return 0.0
    return float(1.0 - entropy / max_entropy)


def concentration_ratio(distribution: np.ndarray, top_n: int = 2) -> float:
    """
    Fraction of total interaction weight concentrated in the *top_n* categories.

    Parameters
    ----------
    distribution : np.ndarray
        Category probability distribution.
    top_n : int
        Number of top categories to sum.

    Returns
    -------
    float  in [0, 1].
    """
    sorted_dist = np.sort(distribution)[::-1]
    return float(np.sum(sorted_dist[:top_n]))


class FilterBubbleDetector:
    """
    Analyses individual users and provides aggregate bubble statistics.

    Usage
    -----
    detector = FilterBubbleDetector(interaction_matrix, video_df, categories)
    report = detector.analyze_user(user_id=5)
    stats   = detector.aggregate_stats()
    """

    def __init__(
        self,
        interaction_matrix: np.ndarray,
        video_df: pd.DataFrame,
        categories: List[str],
    ):
        self.interaction_matrix = interaction_matrix
        self.video_df = video_df
        self.categories = categories
        self.num_categories = len(categories)

    def analyze_user(self, user_id: int) -> Dict:
        """
        Return a bubble analysis report for *user_id*.

        Returns
        -------
        dict with keys:
            user_id, distribution, entropy, bubble_score, top2_concentration,
            top_categories
        """
        dist = _category_distribution(
            user_id, self.interaction_matrix, self.video_df, self.categories
        )
        ent = category_entropy(dist)
        score = bubble_score(ent, self.num_categories)
        top2 = concentration_ratio(dist, top_n=2)

        # Identify top categories for human-readable output
        top_cat_indices = np.argsort(dist)[::-1]
        top_categories = [
            (self.categories[i], round(float(dist[i]), 3))
            for i in top_cat_indices
            if dist[i] > 0.01  # only show meaningful categories
        ]

        return {
            "user_id": user_id,
            "distribution": dist,
            "entropy": round(ent, 4),
            "bubble_score": round(score, 4),
            "top2_concentration": round(top2, 4),
            "top_categories": top_categories,
        }

    def aggregate_stats(self) -> Dict:
        """
        Compute bubble statistics across all users.

        Returns
        -------
        dict with keys:
            mean_bubble_score, std_bubble_score, mean_entropy,
            high_bubble_users (list of user_ids with score > 0.7),
            low_bubble_users  (list of user_ids with score < 0.3)
        """
        num_users = self.interaction_matrix.shape[0]
        scores = []
        entropies = []

        for uid in range(num_users):
            dist = _category_distribution(
                uid, self.interaction_matrix, self.video_df, self.categories
            )
            ent = category_entropy(dist)
            sc = bubble_score(ent, self.num_categories)
            scores.append(sc)
            entropies.append(ent)

        scores = np.array(scores)
        entropies = np.array(entropies)

        return {
            "mean_bubble_score": round(float(np.mean(scores)), 4),
            "std_bubble_score": round(float(np.std(scores)), 4),
            "median_bubble_score": round(float(np.median(scores)), 4),
            "mean_entropy": round(float(np.mean(entropies)), 4),
            "high_bubble_users": np.where(scores > 0.7)[0].tolist(),
            "low_bubble_users": np.where(scores < 0.3)[0].tolist(),
            "all_bubble_scores": scores,
        }
