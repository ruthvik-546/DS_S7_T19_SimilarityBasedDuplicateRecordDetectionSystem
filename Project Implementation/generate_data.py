"""
generate_data.py
-----------------
Creates a synthetic "customer records" dataset that mimics a real messy
database: some records are perfectly clean, some are exact duplicates,
and some are "near duplicates" containing typos, formatting differences,
missing values, or reordered name components. This gives the pipeline
realistic ground truth (a duplicate_of column + is_duplicate label) so
that we can train/evaluate the ML-based classifier described in the
abstract.
"""

import random
import string
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

FIRST_NAMES = ["Akhil", "Ruthvik", "Surya", "Priya", "Ananya", "Rahul", "Sneha",
               "Kiran", "Divya", "Arjun", "Meera", "Vikram", "Pooja", "Karthik",
               "Neha", "Sanjay", "Lakshmi", "Rohit", "Swathi", "Manoj"]
LAST_NAMES = ["Reddy", "Rao", "Kumar", "Sharma", "Naidu", "Prasad", "Varma",
              "Chowdary", "Devi", "Iyer", "Menon", "Gupta", "Verma", "Pillai"]
STREETS = ["MG Road", "Banjara Hills", "Jubilee Hills", "Gachibowli",
           "Kondapur", "Madhapur", "Ameerpet", "Kukatpally", "Miyapur", "Uppal"]
CITIES = ["Hyderabad", "Bengaluru", "Chennai", "Vijayawada", "Warangal"]
DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "rediffmail.com"]


def random_phone():
    return "9" + "".join(random.choices(string.digits, k=9))


def make_base_record(rec_id):
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    email = f"{first.lower()}.{last.lower()}{random.randint(1,99)}@{random.choice(DOMAINS)}"
    phone = random_phone()
    address = f"{random.randint(1,999)}, {random.choice(STREETS)}, {random.choice(CITIES)}"
    return {
        "record_id": rec_id,
        "name": name,
        "email": email,
        "phone": phone,
        "address": address,
    }


def introduce_typo(text):
    """Randomly perturb a string: swap chars, drop a char, or add a char."""
    if len(text) < 3:
        return text
    text = list(text)
    op = random.choice(["swap", "drop", "add", "case"])
    i = random.randint(0, len(text) - 2)
    if op == "swap":
        text[i], text[i + 1] = text[i + 1], text[i]
    elif op == "drop":
        del text[i]
    elif op == "add":
        text.insert(i, random.choice(string.ascii_lowercase))
    elif op == "case":
        text[i] = text[i].upper() if text[i].islower() else text[i].lower()
    return "".join(text)


def make_near_duplicate(base):
    """Create a near-duplicate of a base record with realistic noise."""
    rec = dict(base)
    choice = random.random()

    if choice < 0.3:
        # typo in the name
        rec["name"] = introduce_typo(rec["name"])
    elif choice < 0.5:
        # reordered name components ("Last First" instead of "First Last")
        parts = rec["name"].split()
        if len(parts) == 2:
            rec["name"] = f"{parts[1]} {parts[0]}"
    elif choice < 0.65:
        # missing / blank field
        field = random.choice(["phone", "email", "address"])
        rec[field] = ""
    elif choice < 0.8:
        # formatting difference in phone/address
        if random.random() < 0.5:
            rec["phone"] = rec["phone"][:3] + "-" + rec["phone"][3:6] + "-" + rec["phone"][6:]
        else:
            rec["address"] = rec["address"].upper()
    else:
        # email typo
        rec["email"] = introduce_typo(rec["email"])

    return rec


def generate_dataset(n_unique=350, dup_rate=0.35, exact_dup_share=0.4, out_path=None):
    """
    n_unique: number of distinct underlying people
    dup_rate: fraction of unique records that get at least one duplicate injected
    exact_dup_share: of the injected duplicates, what fraction are EXACT copies
                      (the rest are near-duplicates with noise)
    """
    records = []
    rec_id = 1
    base_ids = []

    for _ in range(n_unique):
        base = make_base_record(rec_id)
        base["duplicate_of"] = -1  # -1 = this is an original record
        records.append(base)
        base_ids.append(rec_id)
        rec_id += 1

    n_dupe_sources = int(n_unique * dup_rate)
    dupe_sources = random.sample(base_ids, n_dupe_sources)

    for src_id in dupe_sources:
        base = next(r for r in records if r["record_id"] == src_id)
        n_copies = random.choice([1, 1, 1, 2])  # usually 1 duplicate, sometimes 2
        for _ in range(n_copies):
            if random.random() < exact_dup_share:
                dup = dict(base)
            else:
                dup = make_near_duplicate(base)
            dup["record_id"] = rec_id
            dup["duplicate_of"] = src_id
            records.append(dup)
            rec_id += 1

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle rows

    if out_path:
        df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    df = generate_dataset(out_path="/home/claude/duplicate_detection/data/raw_records.csv")
    print(f"Generated {len(df)} records "
          f"({(df['duplicate_of'] != -1).sum()} injected duplicates).")
    print(df.head(10))
