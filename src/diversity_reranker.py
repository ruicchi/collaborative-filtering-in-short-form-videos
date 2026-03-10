"""
diversity_reranker.py
---------------------
Re-ranks a list of CF-recommended videos to increase category diversity.

Algorithm: **Maximal Marginal Relevance (MMR)**
MMR iteratively builds a recommendation list by at each step choosing the
candidate that best balances:
  - Relevance: the CF score (how much we predict the user will like it)
  - Diversity: how different it is from the items already selected

Formula at each step:
    argmax [ λ * relevance(v) - (1 - λ) * max_similarity(v, selected) ]

where λ (lambda_param) controls the relevance/diversity trade-off:
  λ = 1.0  →  pure relevance (standard CF order)
  λ = 0.0  →  pure diversity
  λ = 0.5  →  balanced (recommended default)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def _item_diversity(
    video_a: int, video_b: int, video_df: pd.DataFrame
) -> float:
    """
    Diversity between two videos. Currently uses category dissimilarity:
      - Same category  → dissimilarity 0 (not diverse)
      - Diff category  → dissimilarity 1 (maximally diverse)

    This binary measure is simple and effective for category-level diversity.
    """
    cat_a = video_df.loc[video_a, "category"]
    cat_b = video_df.loc[video_b, "category"]
    return 0.0 if cat_a == cat_b else 1.0


def mmr_rerank(
    candidates: List[int],
    cf_scores: np.ndarray,
    video_df: pd.DataFrame,
    top_k: int,
    lambda_param: float = 0.5,
) -> List[int]:
    """
    Maximal Marginal Relevance re-ranking.

    Parameters
    ----------
    candidates : List[int]
        Pool of candidate video_ids (already filtered by CF).
    cf_scores : np.ndarray
        CF relevance score for every video (index = video_id).
        Higher = more relevant.
    video_df : pd.DataFrame
        Video metadata; must contain a "category" column.
    top_k : int
        Number of items to return.
    lambda_param : float
        Trade-off weight (0 = pure diversity, 1 = pure relevance).

    Returns
    -------
    List of *top_k* video_ids in re-ranked order.
    """
    if not candidates:
        return []

    # Normalise CF scores to [0, 1] within the candidate set
    candidate_scores = np.array([cf_scores[v] for v in candidates], dtype=np.float64)
    score_min = candidate_scores.min()
    score_max = candidate_scores.max()
    score_range = score_max - score_min
    if score_range > 0:
        normalised = (candidate_scores - score_min) / score_range
    else:
        normalised = np.ones_like(candidate_scores)

    relevance = {v: float(normalised[i]) for i, v in enumerate(candidates)}

    selected: List[int] = []
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        if not selected:
            # First item: pick the one with the highest CF score
            best = max(remaining, key=lambda v: relevance[v])
        else:
            best_score = -np.inf
            best = remaining[0]
            for video in remaining:
                # Maximum similarity to already-selected items
                # (1 - diversity = similarity)
                max_sim = max(
                    1.0 - _item_diversity(video, sel, video_df) for sel in selected
                )
                mmr = lambda_param * relevance[video] - (1 - lambda_param) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best = video

        selected.append(best)
        remaining.remove(best)

    return selected


class DiversityReranker:
    """
    Convenience wrapper around MMR re-ranking.

    Usage
    -----
    reranker = DiversityReranker(video_df, lambda_param=0.5)
    diverse_list = reranker.rerank(candidates, cf_scores, top_k=10)
    """

    def __init__(self, video_df: pd.DataFrame, lambda_param: float = 0.5):
        """
        Parameters
        ----------
        video_df : pd.DataFrame
            Video metadata.
        lambda_param : float
            Relevance/diversity trade-off. Default 0.5 = balanced.
        """
        self.video_df = video_df
        self.lambda_param = lambda_param

    def rerank(
        self, candidates: List[int], cf_scores: np.ndarray, top_k: int
    ) -> List[int]:
        """
        Re-rank *candidates* using MMR.

        Parameters
        ----------
        candidates : List[int]
            Candidate video_ids from CF.
        cf_scores : np.ndarray
            Full score array (one entry per video).
        top_k : int
            Desired recommendation list length.

        Returns
        -------
        List of *top_k* video_ids, diversity-aware.
        """
        return mmr_rerank(
            candidates=candidates,
            cf_scores=cf_scores,
            video_df=self.video_df,
            top_k=top_k,
            lambda_param=self.lambda_param,
        )

    def category_coverage(self, video_ids: List[int]) -> float:
        """
        Fraction of all categories represented in *video_ids*.

        Returns a value in [0, 1]; 1 = all categories covered.
        """
        if not video_ids:
            return 0.0
        unique_cats = self.video_df.loc[video_ids, "category"].nunique()
        total_cats = self.video_df["category"].nunique()
        return unique_cats / total_cats
