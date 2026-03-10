"""
recommender.py
--------------
Main Recommender class that combines all components into a unified pipeline:

    CF Scores → Bubble Detection → Diversity Re-ranking → Exploration Injection

Exploration (epsilon-greedy)
----------------------------
With probability *epsilon*, one slot in the recommendation list is replaced
by a randomly sampled video from an underrepresented category. This:
  - Exposes the user to content outside their filter bubble.
  - Mimics the "wild card" slots many real platforms use.
  - Prevents the model from purely exploiting known preferences.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict

from src.collaborative_filtering import CollaborativeFilter
from src.filter_bubble_detector import FilterBubbleDetector
from src.diversity_reranker import DiversityReranker


class Recommender:
    """
    Full recommendation pipeline combining CF, bubble detection, diversity
    re-ranking, and epsilon-greedy exploration.

    Parameters
    ----------
    num_recommendations : int
        Number of videos to recommend per user.
    diversity_weight : float
        Lambda for MMR re-ranking (0 = pure diversity, 1 = pure CF).
    exploration_rate : float
        Epsilon for epsilon-greedy exploration. 0.2 → ~20% of recommendations
        will be exploratory.
    cf_method : str
        Which CF strategy to use: "user_based" or "item_based".
    n_neighbors : int
        Number of CF neighbours.
    candidate_pool_size : int
        How many raw CF candidates to pass to the re-ranker.
    seed : int
        Random seed for reproducible exploration.
    """

    def __init__(
        self,
        num_recommendations: int = 10,
        diversity_weight: float = 0.5,
        exploration_rate: float = 0.2,
        cf_method: str = "user_based",
        n_neighbors: int = 20,
        candidate_pool_size: int = 50,
        seed: int = 42,
    ):
        self.num_recommendations = num_recommendations
        self.diversity_weight = diversity_weight
        self.exploration_rate = exploration_rate
        self.cf_method = cf_method
        self.n_neighbors = n_neighbors
        self.candidate_pool_size = candidate_pool_size
        self.rng = np.random.default_rng(seed)

        # Set after fit()
        self.cf: Optional[CollaborativeFilter] = None
        self.bubble_detector: Optional[FilterBubbleDetector] = None
        self.reranker: Optional[DiversityReranker] = None
        self.interaction_matrix: Optional[np.ndarray] = None
        self.video_df: Optional[pd.DataFrame] = None
        self.categories: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        interaction_matrix: np.ndarray,
        video_df: pd.DataFrame,
        categories: List[str],
    ) -> "Recommender":
        """
        Train the CF model and initialise supporting components.

        Parameters
        ----------
        interaction_matrix : np.ndarray  (users × videos)
        video_df : pd.DataFrame
        categories : List[str]
        """
        self.interaction_matrix = interaction_matrix
        self.video_df = video_df
        self.categories = categories

        self.cf = CollaborativeFilter(n_neighbors=self.n_neighbors)
        self.cf.fit(interaction_matrix)

        self.bubble_detector = FilterBubbleDetector(
            interaction_matrix, video_df, categories
        )
        self.reranker = DiversityReranker(
            video_df, lambda_param=self.diversity_weight
        )
        return self

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(
        self, user_id: int, apply_diversity: bool = True
    ) -> Dict:
        """
        Generate recommendations for *user_id*.

        Parameters
        ----------
        user_id : int
        apply_diversity : bool
            If True, apply MMR re-ranking and epsilon-greedy exploration.
            If False, return raw CF output (useful for comparison).

        Returns
        -------
        dict with keys:
            user_id, recommendations (List[int]), bubble_analysis (dict),
            method (str)
        """
        assert self.cf is not None, "Call fit() first."

        # 1. Bubble analysis (always computed, even when diversity is off)
        bubble_info = self.bubble_detector.analyze_user(user_id)

        # 2. Get CF scores + candidate pool
        if self.cf_method == "item_based":
            cf_scores = self.cf.get_item_based_scores(user_id, exclude_seen=True)
        else:
            cf_scores = self.cf.get_user_based_scores(user_id, exclude_seen=True)

        # Sort all finite-scored items to get candidate pool
        finite_mask = np.isfinite(cf_scores)
        candidate_indices = np.where(finite_mask)[0]
        sorted_by_score = candidate_indices[
            np.argsort(cf_scores[candidate_indices])[::-1]
        ]
        candidates = sorted_by_score[: self.candidate_pool_size].tolist()

        if not apply_diversity:
            # Pure CF: just return top-K from candidates
            recs = candidates[: self.num_recommendations]
            return {
                "user_id": user_id,
                "recommendations": recs,
                "bubble_analysis": bubble_info,
                "method": "standard_cf",
            }

        # 3. MMR Diversity Re-ranking
        diverse_recs = self.reranker.rerank(
            candidates=candidates,
            cf_scores=cf_scores,
            top_k=self.num_recommendations,
        )

        # 4. Epsilon-greedy exploration
        diverse_recs = self._apply_exploration(user_id, diverse_recs, bubble_info)

        return {
            "user_id": user_id,
            "recommendations": diverse_recs,
            "bubble_analysis": bubble_info,
            "method": "diversity_aware_cf",
        }

    # ------------------------------------------------------------------
    # Epsilon-greedy exploration
    # ------------------------------------------------------------------

    def _apply_exploration(
        self,
        user_id: int,
        recommendations: List[int],
        bubble_info: Dict,
    ) -> List[int]:
        """
        With probability *exploration_rate*, replace the last recommendation
        with a random video from an underrepresented category.

        Videos the user has already seen are excluded from exploration too.
        """
        if self.rng.random() > self.exploration_rate or not recommendations:
            return recommendations

        # Identify underrepresented categories (bottom half by user interaction)
        user_dist = bubble_info["distribution"]
        sorted_cat_indices = np.argsort(user_dist)  # ascending = least-seen first
        underrepresented = [
            self.categories[i]
            for i in sorted_cat_indices[: max(1, len(self.categories) // 2)]
        ]

        # Find unseen videos from underrepresented categories
        seen_videos = set(np.where(self.interaction_matrix[user_id] != 0)[0].tolist())
        already_recommended = set(recommendations)

        exploration_pool = self.video_df[
            self.video_df["category"].isin(underrepresented)
        ]["video_id"].values

        eligible = [
            v for v in exploration_pool
            if v not in seen_videos and v not in already_recommended
        ]

        if not eligible:
            return recommendations  # Nothing new to explore

        exploration_video = int(self.rng.choice(eligible))
        # Replace the last slot with the exploratory video
        recs = list(recommendations[:-1]) + [exploration_video]
        return recs
