"""
Generate Realistic Amazon-style Product Review Dataset

This creates synthetic but realistic customer reviews for training.
In production, you'd use real Reliance Retail review data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Product categories
categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports', 'Beauty']

# Review templates for different ratings
positive_reviews = [
    "Excellent product! Highly recommended.",
    "Amazing quality, worth every penny.",
    "Great purchase, very satisfied.",
    "Loved it! Will buy again.",
    "Perfect product, fast delivery.",
    "Outstanding quality and service.",
    "Exceeded my expectations completely.",
    "Best purchase I've made in months.",
    "Fantastic product, great value for money.",
    "Wonderful experience, highly recommended."
]

negative_reviews = [
    "Very disappointed with the quality.",
    "Waste of money, not as described.",
    "Poor quality, broke after one use.",
    "Terrible product, do not buy.",
    "Completely unsatisfied, want refund.",
    "Bad quality, not worth the price.",
    "Horrible experience, never buying again.",
    "Cheap material, very poor quality.",
    "Not as advertised, very disappointed.",
    "Worst purchase ever, total waste."
]

neutral_reviews = [
    "It's okay, nothing special.",
    "Average product, does the job.",
    "Decent but could be better.",
    "Not bad but not great either.",
    "Acceptable for the price."
]

def generate_review(rating, category):
    """Generate review text based on rating"""
    if rating in [1, 2]:
        base = np.random.choice(negative_reviews)
        # Add specific complaints
        complaints = [
            f"The {category.lower()} quality is poor.",
            "Delivery was late.",
            "Product arrived damaged.",
            "Not as shown in pictures.",
            "Customer service was unhelpful."
        ]
        return base + " " + np.random.choice(complaints)
    
    elif rating in [4, 5]:
        base = np.random.choice(positive_reviews)
        # Add specific praise
        praise = [
            f"The {category.lower()} quality is excellent.",
            "Fast delivery!",
            "Exactly as described.",
            "Great packaging.",
            "Customer service was helpful."
        ]
        return base + " " + np.random.choice(praise)
    
    else:  # rating 3
        return np.random.choice(neutral_reviews)

# Generate dataset
print("Generating Amazon-style review dataset...")

n_reviews = 5000
data = []

for i in range(n_reviews):
    # Skew towards positive reviews (realistic - satisfied customers review more)
    rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.15, 0.35, 0.35])
    
    category = np.random.choice(categories)
    review_text = generate_review(rating, category)
    
    # Add some variety
    verified = np.random.choice([True, False], p=[0.8, 0.2])
    helpful_votes = int(np.random.exponential(2)) if rating in [1, 2, 5] else 0
    
    # Random date in last year
    date = datetime.now() - timedelta(days=np.random.randint(1, 365))
    
    data.append({
        'review_id': f'R{i:05d}',
        'review_text': review_text,
        'rating': rating,
        'category': category,
        'verified_purchase': verified,
        'helpful_votes': helpful_votes,
        'review_date': date.strftime('%Y-%m-%d')
    })

# Create DataFrame
df = pd.DataFrame(data)

# Create sentiment label
df['sentiment'] = df['rating'].apply(lambda x: 'Positive' if x >= 4 else ('Negative' if x <= 2 else 'Neutral'))

print(f"\n✅ Generated {len(df)} reviews")
print(f"\nRating Distribution:")
print(df['rating'].value_counts().sort_index())
print(f"\nSentiment Distribution:")
print(df['sentiment'].value_counts())

# Save
df.to_csv('/home/claude/sentiment-analysis/data/reviews_raw.csv', index=False)
print(f"\n💾 Saved to: data/reviews_raw.csv")

# Show sample
print("\n📋 Sample Reviews:")
print(df[['review_text', 'rating', 'sentiment']].head(10))
