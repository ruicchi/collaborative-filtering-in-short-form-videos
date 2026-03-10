"""
collaborative_filtering.py
--------------------------
Core Collaborative Filtering (CF) engine.

Two strategies are implemented:
1. **User-Based CF** — Find users who are similar to the target user, then aggregate
   their ratings to predict scores for unseen videos.
2. **Item-Based CF** — Find videos that are similar to videos the user already likes,
   then predict scores for candidate videos.

Both strategies use cosine similarity, which is fast and works well with sparse data.
"""

import numpy as np
from typing import List, Optional


def cosine_similarity_matrix(matrix: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """
    Compute pairwise cosine similarity for every row in *matrix*.

    Parameters
    ----------
    matrix : np.ndarray  (N × M)
        Each row is a vector (e.g. a user's interaction history).
    epsilon : float
        Small value added to norms to avoid division by zero.

    Returns
    -------
    sim : np.ndarray  (N × N)
        sim[i, j] = cosine similarity between row i and row j.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + epsilon
    normalized = matrix / norms
    return normalized @ normalized.T


class CollaborativeFilter:
    """
    Unified Collaborative Filtering model supporting both user-based and item-based CF.

    Usage
    -----
    cf = CollaborativeFilter()
    cf.fit(interaction_matrix)
    scores = cf.predict_user_based(user_id, top_k=20)
    """

    def __init__(self, n_neighbors: int = 20):
        """
        Parameters
        ----------
        n_neighbors : int
            Number of nearest neighbors used when aggregating predictions.
        """
        self.n_neighbors = n_neighbors
        self.interaction_matrix: Optional[np.ndarray] = None
        self.user_similarity: Optional[np.ndarray] = None
        self.item_similarity: Optional[np.ndarray] = None
        self.num_users: int = 0
        self.num_items: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, interaction_matrix: np.ndarray) -> "CollaborativeFilter":
        """
        Pre-compute user-user and item-item similarity matrices.

        Parameters
        ----------
        interaction_matrix : np.ndarray  (num_users × num_items)
        """
        self.interaction_matrix = interaction_matrix.astype(np.float32)
        self.num_users, self.num_items = interaction_matrix.shape

        # User-user similarities: rows are users
        self.user_similarity = cosine_similarity_matrix(self.interaction_matrix)
        # Item-item similarities: rows are items (transpose first)
        self.item_similarity = cosine_similarity_matrix(self.interaction_matrix.T)

        return self

    # ------------------------------------------------------------------
    # User-Based CF
    # ------------------------------------------------------------------

    def predict_user_based(
        self, user_id: int, top_k: int = 20, exclude_seen: bool = True
    ) -> List[int]:
        """
        Recommend *top_k* unseen videos for *user_id* using user-based CF.

        The score for each candidate video is the weighted average of neighbour
        ratings, where the weights are the cosine similarities.

        Parameters
        ----------
        user_id : int
        top_k : int
            Number of recommendations to return.
        exclude_seen : bool
            If True, videos the user has already interacted with are excluded.

        Returns
        -------
        List of video_ids sorted by predicted score (descending).
        """
        assert self.interaction_matrix is not None, "Call fit() first."

        # Identify top-N neighbours (excluding the user themselves)
        similarities = self.user_similarity[user_id].copy()
        similarities[user_id] = 0.0  # exclude self
        neighbor_ids = np.argsort(similarities)[::-1][: self.n_neighbors]
        neighbor_sims = similarities[neighbor_ids]

        # Weighted sum of neighbour ratings
        neighbor_ratings = self.interaction_matrix[neighbor_ids]  # (K × items)
        sim_weights = neighbor_sims[:, np.newaxis]  # (K × 1)
        sim_sum = np.sum(np.abs(neighbor_sims)) + 1e-9
        scores = np.sum(sim_weights * neighbor_ratings, axis=0) / sim_sum

        # Mask already-seen items
        if exclude_seen:
            seen_mask = self.interaction_matrix[user_id] != 0
            scores[seen_mask] = -np.inf

        top_items = np.argsort(scores)[::-1][:top_k]
        return top_items.tolist()

    def get_user_based_scores(
        self, user_id: int, exclude_seen: bool = True
    ) -> np.ndarray:
        """
        Return a score array (one value per item) using user-based CF.
        Seen items are set to -inf when *exclude_seen* is True.
        """
        assert self.interaction_matrix is not None, "Call fit() first."
        similarities = self.user_similarity[user_id].copy()
        similarities[user_id] = 0.0
        neighbor_ids = np.argsort(similarities)[::-1][: self.n_neighbors]
        neighbor_sims = similarities[neighbor_ids]
        neighbor_ratings = self.interaction_matrix[neighbor_ids]
        sim_weights = neighbor_sims[:, np.newaxis]
        sim_sum = np.sum(np.abs(neighbor_sims)) + 1e-9
        scores = np.sum(sim_weights * neighbor_ratings, axis=0) / sim_sum

        if exclude_seen:
            seen_mask = self.interaction_matrix[user_id] != 0
            scores[seen_mask] = -np.inf

        return scores

    # ------------------------------------------------------------------
    # Item-Based CF
    # ------------------------------------------------------------------

    def predict_item_based(
        self, user_id: int, top_k: int = 20, exclude_seen: bool = True
    ) -> List[int]:
        """
        Recommend *top_k* unseen videos using item-based CF.

        For each candidate item, the score is the weighted average of the
        user's existing ratings for similar items.

        Parameters
        ----------
        user_id : int
        top_k : int
        exclude_seen : bool

        Returns
        -------
        List of video_ids sorted by predicted score (descending).
        """
        assert self.interaction_matrix is not None, "Call fit() first."
        scores = self.get_item_based_scores(user_id, exclude_seen=exclude_seen)
        top_items = np.argsort(scores)[::-1][:top_k]
        return top_items.tolist()

    def get_item_based_scores(
        self, user_id: int, exclude_seen: bool = True
    ) -> np.ndarray:
        """Return item-based CF score array for *user_id*."""
        assert self.interaction_matrix is not None, "Call fit() first."
        user_ratings = self.interaction_matrix[user_id]  # (items,)
        # Only items the user has rated contribute to predictions
        rated_mask = user_ratings != 0
        if not np.any(rated_mask):
            return np.zeros(self.num_items, dtype=np.float32)

        # item_similarity[i, j] = similarity between item i and item j
        # score[j] = sum over rated i of (sim(i,j) * rating[i]) / sum(|sim(i,j)|)
        rated_ratings = user_ratings[rated_mask]           # (R,)
        # sim_matrix_t[i, j] = similarity between rated item i and candidate item j
        sim_matrix_t = self.item_similarity[rated_mask, :]  # (R × items)
        scores = (rated_ratings @ sim_matrix_t) / (
            np.sum(np.abs(sim_matrix_t), axis=0) + 1e-9
        )

        if exclude_seen:
            scores[rated_mask] = -np.inf

        return scores
