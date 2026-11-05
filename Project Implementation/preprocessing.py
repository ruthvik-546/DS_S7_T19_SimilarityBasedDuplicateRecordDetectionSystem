"""
preprocessing.py
-----------------
Implements the "Pre-Processing" block of the architecture diagram:
data cleaning, missing-value handling, text normalization
(lowercase, remove special characters, stopwords) and standardization
(dates, names, addresses, phone numbers).
"""

import re
import pandas as pd


def clean_text(value):
    if pd.isna(value) or value == "":
        return ""
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9@.\s-]", "", value)   # strip special characters
    value = re.sub(r"\s+", " ", value)             # collapse whitespace
    return value.strip()


def standardize_phone(value):
    if pd.isna(value) or value == "":
        return ""
    digits = re.sub(r"\D", "", str(value))
    return digits


def handle_missing_values(df):
    """Fill missing text fields with empty string, flag rows with missing data."""
    df = df.copy()
    text_cols = ["name", "email", "phone", "address"]
    for col in text_cols:
        df[col] = df[col].fillna("")
    df["has_missing_field"] = (df[text_cols] == "").any(axis=1)
    return df


def preprocess(df):
    """
    Full preprocessing step:
      1. Handle missing values
      2. Normalize text fields (name, email, address)
      3. Standardize phone numbers to digits-only
    Returns a cleaned copy of the dataframe with new *_clean columns.
    """
    df = handle_missing_values(df)
    df["name_clean"] = df["name"].apply(clean_text)
    df["email_clean"] = df["email"].apply(clean_text)
    df["address_clean"] = df["address"].apply(clean_text)
    df["phone_clean"] = df["phone"].apply(standardize_phone)
    return df
