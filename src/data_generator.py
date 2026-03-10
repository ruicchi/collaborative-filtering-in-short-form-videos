"""
data_generator.py
-----------------
Generates synthetic data that mimics short-form video platforms (TikTok, YouTube Shorts, etc.).

Key design choices:
- Users naturally develop "filter bubbles" by preferring 1-3 categories.
- Interaction types mirror real platforms: watch (mild positive), like (strong positive),
  skip (negative).
- Using a random seed makes results fully reproducible.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# Ten content categories representative of short-form video platforms
CATEGORIES = [
    "Comedy", "Dance", "Cooking", "Tech", "Sports",
    "Music", "Education", "Fashion", "Travel", "Gaming",
]

# Interaction strength values
INTERACTION_WEIGHTS = {
    "watch": 1.0,
    "like": 2.0,
    "skip": -0.5,
}


def generate_data(
    num_users: int = 200,
    num_videos: int = 500,
    num_categories: int = 10,
    interactions_per_user: int = 50,
    preferred_category_prob: float = 0.75,
    seed: int = 42,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Generate synthetic user-video interaction data.

    Parameters
    ----------
    num_users : int
        Number of users to simulate.
    num_videos : int
        Number of videos in the catalogue.
    num_categories : int
        Number of content categories (uses first N from CATEGORIES list).
    interactions_per_user : int
        Approximate number of interactions each user generates.
    preferred_category_prob : float
        Probability that a user's interaction is with one of their preferred categories.
        Higher values create stronger filter bubbles.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    interaction_matrix : np.ndarray  (num_users × num_videos)
        Each cell holds the interaction strength (0 = no interaction).
    user_df : pd.DataFrame
        User metadata including preferred categories.
    video_df : pd.DataFrame
        Video metadata including category.
    categories : List[str]
        Category names used.
    """
    rng = np.random.default_rng(seed)
    categories = CATEGORIES[:num_categories]

    # --- Videos -----------------------------------------------------------------
    # Assign each video to exactly one category (roughly uniform distribution)
    video_categories = rng.choice(categories, size=num_videos)
    video_df = pd.DataFrame({
        "video_id": range(num_videos),
        "category": video_categories,
    })

    # Pre-compute per-category video indices for fast sampling
    cat_to_videos: Dict[str, np.ndarray] = {
        cat: video_df[video_df["category"] == cat]["video_id"].values
        for cat in categories
    }

    # --- Users ------------------------------------------------------------------
    # Each user prefers 1–3 categories (their "filter bubble")
    num_preferred = rng.integers(1, 4, size=num_users)  # 1, 2, or 3
    user_preferred_cats: List[List[str]] = [
        list(rng.choice(categories, size=int(n), replace=False))
        for n in num_preferred
    ]
    user_df = pd.DataFrame({
        "user_id": range(num_users),
        "preferred_categories": user_preferred_cats,
    })

    # --- Interactions -----------------------------------------------------------
    interaction_matrix = np.zeros((num_users, num_videos), dtype=np.float32)

    for user_id in range(num_users):
        preferred = user_preferred_cats[user_id]
        non_preferred = [c for c in categories if c not in preferred]

        for _ in range(interactions_per_user):
            # Decide whether this interaction is with a preferred or other category
            if rng.random() < preferred_category_prob or not non_preferred:
                cat = rng.choice(preferred)
            else:
                cat = rng.choice(non_preferred)

            videos_in_cat = cat_to_videos[cat]
            if len(videos_in_cat) == 0:
                continue
            video_id = int(rng.choice(videos_in_cat))

            # Assign interaction type; preferred-category videos get more likes
            if cat in preferred:
                interaction_type = rng.choice(
                    ["watch", "like", "skip"],
                    p=[0.5, 0.4, 0.1],
                )
            else:
                interaction_type = rng.choice(
                    ["watch", "like", "skip"],
                    p=[0.55, 0.15, 0.30],
                )

            weight = INTERACTION_WEIGHTS[interaction_type]
            # Accumulate weight (a video can be seen multiple times)
            interaction_matrix[user_id, video_id] += weight

    return interaction_matrix, user_df, video_df, categories
