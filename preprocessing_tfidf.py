# src/preprocessing_tfidf.py

import re
import unicodedata
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TfidfPreprocessor:
    
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        self.contractions = {
            r"what's": "what is",
            r"\'s": " is",
            r"\'ve": " have",
            r"can't": "cannot",
            r"n't": " not",
            r"i'm": "i am",
            r"\'re": " are",
            r"\'d": " would",
            r"\'ll": " will",
            r"won't": "will not",
            r"don't": "do not",
            r"doesn't": "does not",
            r"didn't": "did not",
            r"hasn't": "has not",
            r"haven't": "have not",
            r"hadn't": "had not",
            r"wouldn't": "would not",
            r"shouldn't": "should not",
            r"couldn't": "could not"
        }

    def clean_text(self, text: str) -> str:
        
        text = str(text).lower()

        # Normalize unicode (remove emojis, weird chars)
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()

        # Remove repeated full words (wordwordword → word)
        text = re.sub(r'\b(\w{3,})\b(?:\1\b)+', r'\1', text)

        # Normalize laughter (hahahahaha → haha)
        text = re.sub(r'(ha){3,}', r'ha', text)

        # Reduce character repetition (looooool → loool)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # Expand contractions
        for pattern, replacement in self.contractions.items():
            text = re.sub(pattern, replacement, text)

        # Remove URLs and emails
        text = re.sub(r"http\S+|www\S+|\S+@\S+", "", text)

        # Keep only alphabets
        text = re.sub(r"[^a-z\s]", " ", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Remove stopwords + lemmatize
        words = [
            self.lemmatizer.lemmatize(word)
            for word in text.split()
            if word not in self.stop_words and 2 < len(word) < 22
        ]

        return " ".join(words)

    def transform(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        
        df = df.copy()
        df["clean_text"] = df[text_column].astype(str).apply(self.clean_text)
        df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
        return df

