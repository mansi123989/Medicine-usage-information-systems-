# utils.py - Shared Functions
import re
import pandas as pd

def clean_text(text):
    """Clean text - matches training code exactly"""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

import pandas as pd

def text_cleaner(x):
    if isinstance(x, list):
        x = pd.Series(x)
    elif isinstance(x, str):
        x = pd.Series([x])

    return x.apply(clean_text)