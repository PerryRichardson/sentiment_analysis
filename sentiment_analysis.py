"""Sentiment analysis of Amazon product reviews.

This script reads a dataset of Amazon product reviews and predicts
the sentiment of each review (positive, negative, or neutral) using
spaCy together with the spacytextblob extension. Each review is
scored on a polarity scale from -1 (very negative) to +1 (very
positive), and that score is mapped to a sentiment label.

References:
- https://www.expert.ai/blog/natural-language-processing/ -> Natural Language Processing (NLP) and sentiment analysis
- https://www.expert.ai/blog/natural-language-processing-examples/? -> Examples of NLP applications
"""

import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load the Amazon product reviews into a DataFrame.
reviews_df = pd.read_csv(
    "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
)

# Confirm the size of the dataset and the exact column names.
print("Shape:", reviews_df.shape)
print("Columns:", reviews_df.columns.tolist())

# Preprocessing:
# Drop reviews with no text, then keep only the review column.
clean_reviews = reviews_df.dropna(subset=["reviews.text"])
clean_reviews = clean_reviews["reviews.text"]
print("Reviews after cleaning:", len(clean_reviews))

# Load the medium-sized English model and add the spacytextblob component.
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("spacytextblob")

# Quick test of the sentiment analysis on a sample review.
doc = nlp("This product is absolutely fantastic, I love it.")
print("Polarity:", doc._.blob.polarity)