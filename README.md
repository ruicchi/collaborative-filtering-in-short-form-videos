# Collaborative Filtering to Mitigate Filter Bubbles in Short-Form Videos

> [!IMPORTANT]
> This repository contains only a proof of concept for a university research title defense. It was created for demonstration purposes and is not actively maintained or intended for production use.

A simple, fully-working Python system that detects and mitigates **filter bubbles** in short-form video platforms (TikTok, YouTube Shorts, Instagram Reels) using collaborative filtering and diversity-aware re-ranking.

---

## The Problem: Filter Bubbles

Short-form video platforms use engagement-optimised recommendation algorithms.
Over time these algorithms learn that users click more when shown content that is
similar to what they already like — which gradually **narrows the content pool**
until the user only ever sees one or two topics. This is a filter bubble.

Effects of filter bubbles:
- Reduced exposure to new ideas, perspectives, and content types.
- Reinforcement of existing biases and beliefs.
- Decreased platform value for users who do not realise they are missing content.

## The Solution

This project demonstrates how **Collaborative Filtering (CF)** can be combined
with **diversity-aware re-ranking** to break filter bubbles while still providing
personalised, relevant recommendations.

Key techniques used:

| Technique | Purpose |
|---|---|
| User-Based CF | Find similar users, predict what an unseen video would score |
| Item-Based CF | Find similar videos, predict relevance based on past watches |
| Shannon Entropy | Quantify how "bubbled" a user is (low entropy = narrow bubble) |
| Maximal Marginal Relevance (MMR) | Re-rank candidates to balance relevance with category diversity |
| Epsilon-Greedy Exploration | Inject random content from underrepresented categories |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│                     (demo pipeline)                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
          ┌─────────────────┼──────────────────┐
          │                 │                  │
    ┌─────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │   data_    │   │collaborative│   │   filter_   │
    │ generator  │   │ _filtering  │   │  bubble_    │
    │            │   │             │   │  detector   │
    └────────────┘   └──────┬──────┘   └──────┬──────┘
                            │                 │
                     ┌──────▼─────────────────▼──────┐
                     │         recommender.py         │
                     │  CF scores → bubble analysis   │
                     │  → MMR re-ranking → exploration│
                     └──────────────┬────────────────┘
                                    │
                             ┌──────▼──────┐
                             │ evaluation  │
                             │             │
                             └─────────────┘
```

### Data Flow

```
Raw interactions
      │
      ▼
Interaction Matrix  ──►  Collaborative Filtering  ──►  CF Score Array
(users × videos)         (cosine similarity)
                                                              │
                                                  ┌───────────▼───────────┐
                                                  │  Candidate Pool (top 60)│
                                                  └───────────┬───────────┘
                                                              │
                                              ┌───────────────▼─────────────┐
                                              │   Diversity Re-ranking (MMR) │
                                              │   balance relevance + variety │
                                              └───────────────┬─────────────┘
                                                              │
                                              ┌───────────────▼─────────────┐
                                              │  Epsilon-Greedy Exploration  │
                                              │  inject underrepresented cats│
                                              └───────────────┬─────────────┘
                                                              │
                                                     Final Recommendations
```

---

## Project Structure

```
collaborative-filtering-in-short-form-videos/
├── README.md
├── requirements.txt
├── main.py                        # Entry point — run this!
└── src/
    ├── __init__.py
    ├── data_generator.py          # Synthetic data (users, videos, interactions)
    ├── collaborative_filtering.py # CF engine (user-based, item-based, cosine sim)
    ├── filter_bubble_detector.py  # Bubble metrics (entropy, bubble score, concentration)
    ├── diversity_reranker.py      # MMR-based diversity re-ranking
    ├── recommender.py             # Main pipeline: CF + bubble + diversity + exploration
    ├── evaluation.py              # Metrics: coverage, ILD, novelty, bubble reduction
    └── utils.py                   # Helper functions
```

---

## Setup

**Requirements:** Python 3.8+

```bash
# Clone the repository
git clone https://github.com/ruicchi/collaborative-filtering-in-short-form-videos.git
cd collaborative-filtering-in-short-form-videos

# Install dependencies
pip install -r requirements.txt

# Run the demo
python main.py
```

---

## Usage

Running `python main.py` executes the full pipeline and prints:

1. Synthetic data statistics
2. Bubble analysis across all 200 simulated users
3. Per-user recommendation comparison (Standard CF vs. Diversity-Aware CF)
4. Evaluation table with key metrics

### Sample Output

```
=================================================================
  Collaborative Filtering to Mitigate Filter Bubbles
  Short-Form Video Platform Demo
=================================================================

[1] Generating synthetic data ...
    Users    : 200   |   Videos : 500
    Categories: Comedy, Dance, Cooking, Tech, Sports, Music, Education, Fashion, Travel, Gaming

[3] Filter Bubble Analysis (all users) ...
    Mean bubble score  : 0.462
    High-bubble users (score > 0.7): 27

[4] Recommendations for Sample Users ...

  User 0  (Preferred: Music, Bubble score: 0.621)

  Standard CF              |  Diversity-Aware CF
  -------------------------|------------------------
  1. Video 114  [Music]    |  1. Video 114  [Music]
  2. Video 350  [Music]    |  2. Video 341  [Cooking]
  3. Video 125  [Music]    |  3. Video 268  [Education]
  ...                      |  ...
  Unique categories: 1/10  |  Unique categories: 9/10

=================================================================
  EVALUATION RESULTS  (50 users)
=================================================================
Metric                          Standard CF     Diverse CF
-----------------------------------------------------------------
  Category Coverage                  0.1540        0.7800 ↑
  Intra-List Diversity               0.1773        0.9062 ↑
  Post-Rec Bubble Score              0.5191        0.3858 ↓
-----------------------------------------------------------------
  Avg Bubble Reduction               0.1333
=================================================================
```

---

## Key Metrics Explained

| Metric | What it measures | Better = |
|---|---|---|
| **Bubble Score** | 0-1; how concentrated a user's history is in one category. 1 = fully bubbled | Lower |
| **Category Entropy** | Shannon entropy of category distribution. Low entropy = narrow bubble | Higher |
| **Category Coverage** | Fraction of all categories in the recommendation list | Higher |
| **Intra-List Diversity (ILD)** | Average pairwise category dissimilarity in the rec list | Higher |
| **Novelty Score** | How "unexpected" / unpopular the recommended videos are | Context-dependent |
| **Bubble Reduction** | Drop in bubble score after diversity-aware recommendations | Higher |

---

## Configuration

Edit the constants at the top of `main.py`:

```python
NUM_USERS = 200            # Number of simulated users
NUM_VIDEOS = 500           # Number of videos in the catalogue
NUM_CATEGORIES = 10        # Number of content categories
INTERACTIONS_PER_USER = 50 # Interactions generated per user
NUM_RECOMMENDATIONS = 10   # Recommendations per user
DIVERSITY_WEIGHT = 0.5     # λ for MMR: 0=pure diversity, 1=pure CF relevance
EXPLORATION_RATE = 0.2     # ε for exploration: probability of injecting a wild-card
```

Or configure the `Recommender` programmatically:

```python
from src.recommender import Recommender

rec = Recommender(
    num_recommendations=10,
    diversity_weight=0.5,   # Tune λ
    exploration_rate=0.2,   # Tune ε
    cf_method="user_based", # or "item_based"
    n_neighbors=20,
)
rec.fit(interaction_matrix, video_df, categories)

result = rec.recommend(user_id=42, apply_diversity=True)
print(result["recommendations"])
print(result["bubble_analysis"])
```

---

## How It Works

### 1. Collaborative Filtering

Cosine similarity is computed between every pair of users:

```
sim(u, v) = (u · v) / (||u|| × ||v||)
```

For each target user, the top-K most similar users are found (neighbours).
Their ratings are aggregated (weighted by similarity) to produce a predicted
score for each unseen video.

### 2. Filter Bubble Detection

Shannon entropy of a user's category distribution:

```
H(X) = -Σ p(c) × log₂(p(c))
```

This is normalised to a **Bubble Score**:

```
bubble_score = 1 - H(X) / log₂(num_categories)
```

A score of `1.0` means the user only ever watches one category.
A score of `0.0` means perfectly uniform distribution across all categories.

### 3. Diversity Re-ranking (MMR)

At each step, the next recommendation is selected by:

```
argmax [ λ × relevance(v) - (1-λ) × max_sim(v, already_selected) ]
```

This greedily builds a list that is both relevant **and** diverse.

### 4. Epsilon-Greedy Exploration

With probability ε, the last recommendation slot is replaced by a randomly
sampled video from an underrepresented category (one of the bottom 50% by
the user's interaction weight). This ensures long-tail categories get
occasional exposure.

---

## References

- Resnick, P. et al. (1994). *GroupLens: An open architecture for collaborative filtering.* CSCW.
- Carbonell, J. & Goldstein, J. (1998). *The use of MMR, diversity-based reranking for reordering documents.* SIGIR.
- Pariser, E. (2011). *The Filter Bubble: What the Internet is Hiding from You.* Penguin.
- Nguyen, T. T. et al. (2014). *Exploring the filter bubble: the effect of using recommender systems on content diversity.* WWW.
