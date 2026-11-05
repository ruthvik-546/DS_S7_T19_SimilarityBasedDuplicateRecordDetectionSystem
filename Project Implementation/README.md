# Similarity-Based Duplicate Record Detection System

A working implementation of the project described in the abstract (Data
Science 24AD3107, KLH). It detects exact and near-duplicate records in a
dataset (names, addresses, emails, phone numbers) using string similarity,
TF-IDF/cosine/Jaccard similarity, and ML classification — then outputs a
duplicate report, a cleaned deduplicated dataset, and evaluation charts.

## How it works (matches the architecture diagram)

1. **Input Dataset** — `src/generate_data.py` builds a realistic synthetic
   "customer records" CSV with injected exact duplicates and near-duplicates
   (typos, reordered names, missing fields, formatting differences). Swap
   this out for `pd.read_csv("your_file.csv")` to run on real data.
2. **Pre-Processing** (`src/preprocessing.py`) — cleans text, normalizes
   case/special characters, standardizes phone numbers, and flags missing
   values.
3. **Data Splitting + Feature Representation** (`src/feature_engineering.py`)
   — generates candidate record pairs using a blocking key (avoids comparing
   every record against every other one), then computes:
   - **Approach 1 — String similarity:** Levenshtein ratio, Jaro-Winkler
   - **Approach 2 — TF-IDF / Cosine / Jaccard:** vectorizes combined text
     fields and computes similarity, plus a weighted **combined score**
4. **Classification / Decision** (`src/models.py`) — trains Logistic
   Regression, Random Forest, and XGBoost on the pair features (70/30
   train/test split), with grid-search hyperparameter tuning and 5-fold
   cross-validation.
5. **Performance Metrics** — accuracy, precision, recall, F1-score per model;
   best model auto-selected by F1.
6. **Output** (`src/main.py`) — writes:
   - `outputs/duplicate_records_report.csv` — flagged duplicate pairs
   - `outputs/cleaned_unique_dataset.csv` — deduplicated dataset (keeps the
     most complete record per duplicate cluster, using union-find)
   - `outputs/model_comparison.png`, `confusion_matrix.png`,
     `similarity_score_distribution.png`, `dataset_size_comparison.png`
   - `outputs/model_performance_metrics.json`

## Setup

```bash
pip install pandas numpy scikit-learn xgboost python-Levenshtein jellyfish matplotlib seaborn
```

## Run

```bash
python3 src/main.py
```

This regenerates the synthetic dataset, runs the full pipeline, and prints a
metrics summary to the console.

## Using your own data

Replace the call to `generate_dataset()` in `src/main.py` with:

```python
df_raw = pd.read_csv("your_file.csv")  # needs columns: record_id, name, email, phone, address
```

If you don't have ground-truth duplicate labels, skip `label_pairs()` and
instead threshold directly on `combined_score` (e.g. `> 0.85`) rather than
training a supervised classifier — or manually label a sample of pairs to
bootstrap training data.

## Project structure

```
duplicate_detection/
├── data/
│   └── raw_records.csv          # generated synthetic input
├── src/
│   ├── generate_data.py         # synthetic data generator
│   ├── preprocessing.py         # cleaning / normalization / standardization
│   ├── feature_engineering.py   # blocking, string + TF-IDF/cosine/Jaccard similarity
│   ├── models.py                # LogReg / RandomForest / XGBoost + tuning + metrics
│   └── main.py                  # orchestrates the full pipeline
├── outputs/                     # generated report, cleaned dataset, charts
└── README.md
```

## Notes for your project report / viva

- The **blocking step** is what makes this scale to large datasets — instead
  of comparing every pair (O(n²)), records are grouped into blocks (by
  sorted name-initials + phone prefix) and only compared within a block,
  which is standard practice in real-world entity resolution/record-linkage
  systems.
- The **combined similarity score** is a weighted blend of Jaro-Winkler
  (name/address), TF-IDF cosine, Jaccard, and exact-match flags for
  phone/email — this mirrors the abstract's "combined similarity score with
  suitable threshold" description, and also feeds into the ML classifier as
  a feature.
- All three models perform well here because the synthetic noise is
  moderate; on a real, messier dataset you should expect to see more
  separation between the models and it's worth experimenting with the
  blocking strategy and feature weights.
