"""
utils.py
--------
Shared helper functions used across the project.
"""

import numpy as np
import pandas as pd
from typing import List, Dict


def print_user_profile(
    user_id: int,
    user_df: pd.DataFrame,
    bubble_info: Dict,
) -> None:
    """Print a summary of a user's profile and bubble status."""
    preferred = user_df.loc[user_id, "preferred_categories"]
    print(f"\n  User {user_id}")
    print(f"    Preferred categories : {', '.join(preferred)}")
    print(f"    Bubble score         : {bubble_info['bubble_score']:.3f}  (1=fully bubbled, 0=diverse)")
    print(f"    Category entropy     : {bubble_info['entropy']:.3f} bits")
    print(f"    Top-2 concentration  : {bubble_info['top2_concentration']:.1%}")
    top = bubble_info["top_categories"][:3]
    top_str = ", ".join(f"{cat} ({pct:.1%})" for cat, pct in top)
    print(f"    Top categories       : {top_str}")


def print_recommendations(
    recs: List[int],
    video_df: pd.DataFrame,
    label: str = "Recommendations",
) -> None:
    """Print a labelled list of recommended videos with their categories."""
    print(f"\n  {label}:")
    for rank, vid in enumerate(recs, 1):
        cat = video_df.loc[vid, "category"]
        print(f"    {rank:>2}. Video {vid:>4}  [{cat}]")


def category_distribution_str(
    recs: List[int], video_df: pd.DataFrame
) -> str:
    """Return a compact string showing category counts for a rec list."""
    cats = video_df.loc[recs, "category"].value_counts()
    parts = [f"{cat}: {cnt}" for cat, cnt in cats.items()]
    return ", ".join(parts)


def format_bar(value: float, width: int = 20, fill: str = "█") -> str:
    """Return a simple text progress-bar representation of *value* in [0, 1]."""
    filled = int(round(value * width))
    return fill * filled + "░" * (width - filled)
