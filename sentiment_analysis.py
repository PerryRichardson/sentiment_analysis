"""Sentiment analysis of Amazon product reviews.

This script reads a dataset of Amazon product reviews and predicts
the sentiment of each review (positive, negative, or neutral) using
spaCy together with the spacytextblob extension. Each review is
scored on a polarity scale from -1 (very negative) to +1 (very
positive), and that score is mapped to a sentiment label.
"""

import pandas as pd

# Load the Amazon product reviews into a DataFrame.
reviews_df = pd.read_csv(
    "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
)

# Confirm the size of the dataset and the exact column names.
print("Shape:", reviews_df.shape)
print("Columns:", reviews_df.columns.tolist())