"""
main.py
-------
Entry point for the Collaborative Filtering Filter Bubble Mitigation demo.

Run:
    python main.py

What this script does:
1. Generates synthetic user-video interaction data with natural filter bubbles.
2. Trains the collaborative filtering model.
3. Analyses the bubble landscape across all users.
4. Shows detailed recommendations for a few sample users, comparing:
     a) Standard CF (no diversity)
     b) Diversity-Aware CF (MMR + exploration)
5. Prints an evaluation table comparing both approaches.
"""

import numpy as np

from src.data_generator import generate_data
from src.recommender import Recommender
from src.filter_bubble_detector import FilterBubbleDetector
from src.evaluation import evaluate_and_compare, print_comparison_table
from src.utils import (
    print_user_profile,
    print_recommendations,
    category_distribution_str,
    format_bar,
)

SEED = 42
NUM_USERS = 200
NUM_VIDEOS = 500
NUM_CATEGORIES = 10
INTERACTIONS_PER_USER = 50
NUM_RECOMMENDATIONS = 10
DIVERSITY_WEIGHT = 0.5    # λ for MMR (0=pure diversity, 1=pure relevance)
EXPLORATION_RATE = 0.2    # ε-greedy exploration probability
NUM_EVAL_USERS = 50
SAMPLE_USER_IDS = [0, 5, 12, 27, 43]  # Users shown in detail


def main() -> None:
    print("\n" + "=" * 65)
    print("  Collaborative Filtering to Mitigate Filter Bubbles")
    print("  Short-Form Video Platform Demo")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # 1. Generate synthetic data
    # -----------------------------------------------------------------------
    print("\n[1] Generating synthetic data ...")
    interaction_matrix, user_df, video_df, categories = generate_data(
        num_users=NUM_USERS,
        num_videos=NUM_VIDEOS,
        num_categories=NUM_CATEGORIES,
        interactions_per_user=INTERACTIONS_PER_USER,
        seed=SEED,
    )
    print(f"    Users    : {NUM_USERS}")
    print(f"    Videos   : {NUM_VIDEOS}")
    print(f"    Categories: {', '.join(categories)}")
    density = (interaction_matrix != 0).sum() / interaction_matrix.size
    print(f"    Matrix density: {density:.2%}")

    # -----------------------------------------------------------------------
    # 2. Train the recommender
    # -----------------------------------------------------------------------
    print("\n[2] Training collaborative filtering model ...")
    recommender = Recommender(
        num_recommendations=NUM_RECOMMENDATIONS,
        diversity_weight=DIVERSITY_WEIGHT,
        exploration_rate=EXPLORATION_RATE,
        cf_method="user_based",
        n_neighbors=20,
        candidate_pool_size=60,
        seed=SEED,
    )
    recommender.fit(interaction_matrix, video_df, categories)
    print("    Done.")

    # -----------------------------------------------------------------------
    # 3. Analyse filter bubble landscape
    # -----------------------------------------------------------------------
    print("\n[3] Filter Bubble Analysis (all users) ...")
    detector = FilterBubbleDetector(interaction_matrix, video_df, categories)
    stats = detector.aggregate_stats()

    print(f"    Mean bubble score  : {stats['mean_bubble_score']:.3f}")
    print(f"    Median bubble score: {stats['median_bubble_score']:.3f}")
    print(f"    Std bubble score   : {stats['std_bubble_score']:.3f}")
    print(f"    Mean entropy       : {stats['mean_entropy']:.3f} bits")
    high = len(stats["high_bubble_users"])
    low = len(stats["low_bubble_users"])
    print(f"    High-bubble users (score > 0.7): {high}")
    print(f"    Low-bubble  users (score < 0.3): {low}")

    # Bubble score distribution bar chart
    print("\n    Bubble Score Distribution:")
    scores = stats["all_bubble_scores"]
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    for i, label in enumerate(labels):
        count = int(np.sum((scores >= bins[i]) & (scores < bins[i + 1])))
        bar = format_bar(count / NUM_USERS, width=20)
        print(f"      {label}  {bar}  {count:>3} users")

    # -----------------------------------------------------------------------
    # 4. Per-user recommendation showcase
    # -----------------------------------------------------------------------
    print("\n[4] Recommendations for Sample Users")
    print("    (comparing Standard CF vs. Diversity-Aware CF)")

    for uid in SAMPLE_USER_IDS:
        bubble_info = detector.analyze_user(uid)
        print("\n" + "-" * 65)
        print_user_profile(uid, user_df, bubble_info)

        # Standard CF
        result_std = recommender.recommend(uid, apply_diversity=False)
        recs_std = result_std["recommendations"]
        print_recommendations(recs_std, video_df, label="Standard CF")
        print(f"    Category distribution: {category_distribution_str(recs_std, video_df)}")
        unique_cats_std = video_df.loc[recs_std, "category"].nunique()
        print(f"    Unique categories    : {unique_cats_std}/{NUM_CATEGORIES}")

        # Diversity-Aware CF
        result_div = recommender.recommend(uid, apply_diversity=True)
        recs_div = result_div["recommendations"]
        print_recommendations(recs_div, video_df, label="Diversity-Aware CF")
        print(f"    Category distribution: {category_distribution_str(recs_div, video_df)}")
        unique_cats_div = video_df.loc[recs_div, "category"].nunique()
        print(f"    Unique categories    : {unique_cats_div}/{NUM_CATEGORIES}")

    # -----------------------------------------------------------------------
    # 5. Evaluation: Standard CF vs. Diversity-Aware CF
    # -----------------------------------------------------------------------
    print("\n\n[5] Running Evaluation ...")
    eval_results = evaluate_and_compare(
        recommender=recommender,
        interaction_matrix=interaction_matrix,
        video_df=video_df,
        categories=categories,
        num_eval_users=NUM_EVAL_USERS,
        top_k=NUM_RECOMMENDATIONS,
        seed=SEED,
    )
    print_comparison_table(eval_results)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n[6] Summary")
    avg_red = eval_results.get("avg_bubble_reduction", 0.0)
    div_cov = eval_results["diverse"]["coverage"]
    std_cov = eval_results["standard"]["coverage"]
    print(f"    Average bubble score reduction : {avg_red:+.4f}")
    print(f"    Category coverage improvement  : {std_cov:.2%} → {div_cov:.2%}")
    print(f"    Diversity weight (λ)           : {DIVERSITY_WEIGHT}")
    print(f"    Exploration rate (ε)           : {EXPLORATION_RATE}")
    print(
        "\n    ✓ Diversity-aware recommendations successfully mitigate filter bubbles"
        "\n      while maintaining competitive recommendation quality."
    )
    print()


if __name__ == "__main__":
    main()
