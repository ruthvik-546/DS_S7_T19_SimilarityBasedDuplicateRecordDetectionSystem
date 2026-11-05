"""
feature_engineering.py
-----------------------
Implements the "Feature Representation" and "Similarity Computation"
blocks of the architecture:
  - Blocking / candidate pair generation (so we don't compare every
    record against every other record -- standard practice for
    large datasets).
  - Approach 1: String Similarity (Levenshtein distance, Jaro-Winkler)
  - Approach 2: TF-IDF + Cosine Similarity + Jaccard Similarity
  - A weighted Combined Similarity Score
"""

import itertools
import numpy as np
import pandas as pd
import Levenshtein
import jellyfish
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def blocking_key(row):
    """
    Cheap blocking key so we only compare records that are plausibly
    similar (huge speedup on large datasets, mirrors real-world ER
    pipelines). Uses sorted first-letters of name tokens (robust to
    reordered "First Last" vs "Last First") plus first 3 phone digits.
    """
    tokens = sorted(row["name_clean"].split())
    name_part = "".join(t[0] for t in tokens if t) if tokens else "x"
    phone_part = row["phone_clean"][:3] if row["phone_clean"] else "xxx"
    return f"{name_part}_{phone_part}"


def generate_candidate_pairs(df):
    """Group records into blocks, then take all pairs within each block."""
    df = df.copy()
    df["block_key"] = df.apply(blocking_key, axis=1)

    pairs = []
    for _, group in df.groupby("block_key"):
        ids = group["record_id"].tolist()
        if len(ids) > 1:
            pairs.extend(itertools.combinations(ids, 2))

    # Safety net: also pair every record with a couple of nearest
    # neighbours by name (helps recall when blocking misses noisy typos).
    name_sorted = df.sort_values("name_clean")["record_id"].tolist()
    for i in range(len(name_sorted) - 1):
        pairs.append((name_sorted[i], name_sorted[i + 1]))

    pairs = list({tuple(sorted(p)) for p in pairs})  # dedupe
    return pairs


def jaccard_similarity(a, b):
    set_a, set_b = set(a.split()), set(b.split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def build_pair_features(df, pairs):
    """
    For every candidate pair, compute:
      - Levenshtein ratio & Jaro-Winkler similarity on name/email/address
      - Exact-match flags for phone/email
      - Jaccard similarity on name tokens
    TF-IDF cosine similarity is added afterward (needs the full corpus).
    """
    rec = df.set_index("record_id")
    rows = []

    for id_a, id_b in pairs:
        a, b = rec.loc[id_a], rec.loc[id_b]

        row = {
            "id_a": id_a,
            "id_b": id_b,
            "name_levenshtein": Levenshtein.ratio(a["name_clean"], b["name_clean"]),
            "name_jaro_winkler": jellyfish.jaro_winkler_similarity(a["name_clean"], b["name_clean"]),
            "email_levenshtein": Levenshtein.ratio(a["email_clean"], b["email_clean"]),
            "address_levenshtein": Levenshtein.ratio(a["address_clean"], b["address_clean"]),
            "address_jaro_winkler": jellyfish.jaro_winkler_similarity(a["address_clean"], b["address_clean"]),
            "name_jaccard": jaccard_similarity(a["name_clean"], b["name_clean"]),
            "address_jaccard": jaccard_similarity(a["address_clean"], b["address_clean"]),
            "phone_exact_match": int(a["phone_clean"] != "" and a["phone_clean"] == b["phone_clean"]),
            "email_exact_match": int(a["email_clean"] != "" and a["email_clean"] == b["email_clean"]),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def add_tfidf_cosine(df, pair_df):
    """Vectorize a combined text field (name+address+email) and add cosine similarity per pair."""
    df = df.copy()
    df["combined_text"] = (df["name_clean"] + " " + df["address_clean"] + " " + df["email_clean"]).str.strip()

    vectorizer = TfidfVectorizer(min_df=1)
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
    idx_lookup = {rid: i for i, rid in enumerate(df["record_id"])}

    cos_sims = []
    for _, r in pair_df.iterrows():
        i, j = idx_lookup[r["id_a"]], idx_lookup[r["id_b"]]
        sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0, 0]
        cos_sims.append(sim)

    pair_df = pair_df.copy()
    pair_df["tfidf_cosine"] = cos_sims
    return pair_df


def add_combined_score(pair_df, weights=None):
    """Weighted combined similarity score, per the abstract's 'suitable threshold' approach."""
    if weights is None:
        weights = {
            "name_jaro_winkler": 0.25,
            "tfidf_cosine": 0.25,
            "address_jaro_winkler": 0.15,
            "name_jaccard": 0.10,
            "phone_exact_match": 0.15,
            "email_exact_match": 0.10,
        }
    pair_df = pair_df.copy()
    pair_df["combined_score"] = sum(pair_df[col] * w for col, w in weights.items())
    return pair_df


def label_pairs(pair_df, df):
    """Ground-truth label: 1 if both records belong to the same underlying person."""
    indexed = df.set_index("record_id")
    group_map = {
        rid: (rid if row["duplicate_of"] == -1 else row["duplicate_of"])
        for rid, row in indexed.iterrows()
    }
    # a record's own group id may itself point to a duplicate_of chain of length 1 (by construction)
    pair_df = pair_df.copy()
    pair_df["is_duplicate"] = pair_df.apply(
        lambda r: int(group_map[r["id_a"]] == group_map[r["id_b"]]), axis=1
    )
    return pair_df
