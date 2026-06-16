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
# new df to preserve the original df for later use if needed.
clean_reviews = reviews_df.dropna(subset=["reviews.text"])
clean_reviews = clean_reviews["reviews.text"]
print("Reviews after cleaning:", len(clean_reviews))

# Load the medium-sized English model and add the spacytextblob component.
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("spacytextblob")

# Quick test of the sentiment analysis on a sample review.
doc = nlp("This product is absolutely fantastic, I love it.")
print("Polarity:", doc._.blob.polarity)

# Classify a review as positive, negative, or neutral from its polarity.
def classify_sentiment(polarity):
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"


# Sentiment function
# Classify a review as positive, negative, or neutral from its polarity.
def classify_sentiment(review):
    """Predict the sentiment of a single product review.

    Runs the review through the spaCy pipeline, reads its polarity
    score, and returns "positive", "negative", or "neutral". A score
    above 0.1 counts as positive and below -0.1 as negative; the band
    between them is treated as neutral.
    """
    doc = nlp(review)
    polarity = doc._.blob.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"
    
# Apply the sentiment function to test reviews and print the results.
test_reviews = [
    "This product is absolutely fantastic, I love it.",
    "This product is terrible, I hate it.",
    "This product is okay, not great but not bad either.",
]

for review in test_reviews:
    sentiment = classify_sentiment(review)
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment}")
    print()

# Preprocessing function
# To clean text and strip stop words.
def preprocess_review(review):
    """Preprocess a review by removing stop words and punctuation."""
    doc = nlp(review)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct
    ]
    return " ".join(tokens)

# Apply preprocessing to the test reviews.
preprocessed_reviews = [preprocess_review(review) for review in test_reviews]

# Classify the sentiment of each preprocessed review.
for review in preprocessed_reviews:
    sentiment = classify_sentiment(review)
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment}")
    print()

# Note: Stop-word removal strips negation, and negation is central to sentiment.

