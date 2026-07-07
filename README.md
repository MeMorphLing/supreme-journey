# Content-Based Filtering Recommender System (Neural Network)

A two-tower neural network that learns movie and user embeddings jointly, then scores a user–movie pair by the dot product of their embeddings — the neural-net evolution of classical content-based filtering. Built as the Week 2 practice lab for **DeepLearning.AI's Machine Learning Specialization, Course 3 (Unsupervised Learning, Recommenders, Reinforcement Learning)**.

---

## Repository Structure

```
.
├── C3_W2_RecSysNN_Assignment.ipynb   # main notebook — model, training, predictions, similarity search
├── recsysNN_utils.py                 # data loading + pretty-print helpers
├── public_tests.py                   # unit tests for the two graded exercises
├── data/                             # MovieLens-derived CSVs + pickle (not included — see Data section)
└── images/                           # lab illustrations referenced in the notebook markdown (not included)
```

---

## Problem & Approach

Given a user's per-genre rating history and a movie's year/genre/average-rating profile, predict how that user would rate that movie — then use the model's learned representations to recommend similar movies to each other, independent of any specific user.

The network has two identical-shaped towers:

```
User features (14) ──► Dense(256, relu) ──► Dense(128, relu) ──► Dense(32, linear) ──► L2-normalize ──┐
                                                                                                          ├─► Dot product ──► predicted rating
Item features (16) ──► Dense(256, relu) ──► Dense(128, relu) ──► Dense(32, linear) ──► L2-normalize ──┘
```

Both towers are trained jointly via the dot-product output — no separate loss per tower. L2-normalizing each 32-dim embedding before the dot product turns the score into a bounded similarity measure rather than an unconstrained inner product, which is what makes the resulting item vectors usable for similarity search after training.

Built with the Keras **functional API** (not `Sequential`) to allow the two towers to merge into a single dual-input model.

---

## Dataset

Derived from **MovieLens ml-latest-small** (F. Maxwell Harper and Joseph A. Konstan, 2015, *ACM TiiS* 5, 4: 19:1–19:19), filtered to movies from 2000+ in popular genres:

- 397 users · 847 movies · 25,521 original ratings
- Expanded to **50,884 training vectors** (some ratings duplicated to boost underrepresented genres)
- 14 genre categories; item features = year + avg rating + one-hot genres; user features = per-genre average rating
- Split 80/20 → **40,707 train / 10,177 test** (`train_test_split`, fixed seed, user/item/target shuffled identically)
- Inputs standardized with `StandardScaler`; target ratings scaled to [-1, 1] with `MinMaxScaler`

`recsysNN_utils.load_data()` expects the following in `./data/`, which are lab-environment files not included in this repo: `content_item_train.csv`, `content_user_train.csv`, `content_y_train.csv`, `content_item_train_header.txt`, `content_user_train_header.txt`, `content_item_vecs.csv`, `content_movie_list.csv`, `content_user_to_genre.pickle`.

---

## Results

| Metric | Value |
|---|---|
| Training loss (MSE), epoch 1 | 0.1232 |
| Training loss (MSE), epoch 30 | 0.0713 |
| Test loss (MSE) | 0.0815 |

Test loss tracking close to the final training loss indicates the model generalizes without substantial overfitting. Predicted ratings for existing users are generally within ~1 point of their actual rating, though the model is a rough predictor for users whose specific ratings diverge sharply from their own genre averages.

---

## Finding Similar Items

Once trained, the item tower alone (`item_NN`) is reused to embed all 847 movies into 32-dim vectors. Pairwise squared distance between movie embeddings:

```
sq_dist(a, b) = Σ (aᵢ - bᵢ)²
```

is computed for every movie pair, and for each movie the nearest neighbor (excluding itself, via a masked diagonal) becomes its "similar movie" recommendation — no retraining or user data required, since this runs purely on the learned item embeddings.

---

## Graded Exercises

| Exercise | What it checks |
|---|---|
| **Exercise 1** — `user_NN` / `item_NN` architecture | 3-layer Sequential tower: Dense(256, relu) → Dense(128, relu) → Dense(32, linear) |
| **Exercise 2** — `sq_dist(a, b)` | Squared Euclidean distance between two feature vectors |

Both are verified against `public_tests.py` (`test_tower`, `test_sq_dist`) and pass.

---

## Setup & Running

```bash
pip install numpy pandas tensorflow scikit-learn tabulate
```

Requires the `data/` and `images/` directories from the original lab environment (see Dataset section) placed alongside the notebook. Then open `C3_W2_RecSysNN_Assignment.ipynb` and run top to bottom — training takes a few minutes on CPU (30 epochs, ~4s/epoch on the reference run).

---

## Note

This is a solved version of a graded Coursera/DeepLearning.AI lab, kept here for portfolio and reference purposes. If you're currently taking this course, write your own solution for the `### START CODE HERE ###` blocks before consulting this — per Coursera's Honor Code.

## Acknowledgments

- Lab authored by DeepLearning.AI / Andrew Ng, Machine Learning Specialization, Course 3, Week 2
- Dataset: GroupLens Research, MovieLens ml-latest-small
