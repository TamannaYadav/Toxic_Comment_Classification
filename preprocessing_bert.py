# src/preprocessing_bert.py

import re
import unicodedata
import pandas as pd


class BertPreprocessor:
    

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def clean_text(self, text: str) -> str:
        
        text = str(text)

        # Optional lowercase (use True for bert-base-uncased)
        if self.lowercase:
            text = text.lower()

        # Normalize unicode (removes weird characters, keeps text readable)
        text = unicodedata.normalize("NFKC", text)

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Replace multiple spaces/newlines with single space
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def transform(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        
        df = df.copy()
        df["clean_text"] = df[text_column].astype(str).apply(self.clean_text)
        df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
        return df