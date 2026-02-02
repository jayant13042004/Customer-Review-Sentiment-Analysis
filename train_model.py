"""
SENTIMENT ANALYSIS ML PIPELINE
End-to-End: Data → Preprocessing → Features → Model → Evaluation

This is what you'll walk through in interviews.
"""

import pandas as pd
import numpy as np
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("CUSTOMER REVIEW SENTIMENT ANALYSIS - ML PIPELINE")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n📂 STEP 1: Loading Data...")

df = pd.read_csv('/home/claude/sentiment-analysis/data/reviews_raw.csv')
print(f"Loaded {len(df)} reviews")
print(f"\nOriginal Distribution:")
print(df['sentiment'].value_counts())

# ============================================================================
# STEP 2: DATA PREPARATION
# ============================================================================
print("\n🔧 STEP 2: Data Preparation...")

# Remove neutral reviews (rating = 3) for clearer classification
# WHY: 3-star reviews are ambiguous - "okay", "decent" are hard to classify
df_binary = df[df['sentiment'] != 'Neutral'].copy()

print(f"\nAfter removing Neutral:")
print(f"Positive: {(df_binary['sentiment'] == 'Positive').sum()}")
print(f"Negative: {(df_binary['sentiment'] == 'Negative').sum()}")

# Check class imbalance
pos_ratio = (df_binary['sentiment'] == 'Positive').sum() / len(df_binary)
print(f"\nClass Balance: {pos_ratio*100:.1f}% Positive, {(1-pos_ratio)*100:.1f}% Negative")

if pos_ratio > 0.7 or pos_ratio < 0.3:
    print("⚠️  CLASS IMBALANCE DETECTED - Will handle in evaluation")

# Create binary labels: 1 = Positive, 0 = Negative
df_binary['label'] = (df_binary['sentiment'] == 'Positive').astype(int)

# ============================================================================
# STEP 3: TEXT PREPROCESSING
# ============================================================================
print("\n🧹 STEP 3: Text Preprocessing...")

def preprocess_text(text):
    """
    Clean and normalize text for ML models
    
    WHY each step:
    1. Lowercase - "Great" and "great" should be same
    2. Remove URLs - Not useful for sentiment
    3. Remove numbers - "5 stars" → focus on "stars"
    4. Remove punctuation - "great!" → "great"
    5. Remove extra spaces - Clean up
    
    NOTE: We DON'T remove stopwords initially because words like
    "not", "no", "never" are CRITICAL for sentiment!
    """
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove numbers (optional - depends on use case)
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Apply preprocessing
print("Cleaning review text...")
df_binary['review_clean'] = df_binary['review_text'].apply(preprocess_text)

# Show example
print("\n📝 Example of preprocessing:")
sample_idx = 0
print(f"BEFORE: {df_binary['review_text'].iloc[sample_idx]}")
print(f"AFTER:  {df_binary['review_clean'].iloc[sample_idx]}")

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================
print("\n✂️  STEP 4: Train-Test Split...")

# WHY 80-20 split:
# - Need enough training data (80%) to learn patterns
# - Need enough test data (20%) for reliable evaluation
# - Stratify to maintain class balance in both sets

X = df_binary['review_clean']
y = df_binary['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintain class distribution
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"\nTrain distribution: {y_train.value_counts().to_dict()}")
print(f"Test distribution: {y_test.value_counts().to_dict()}")

# ============================================================================
# STEP 5: FEATURE ENGINEERING - TF-IDF
# ============================================================================
print("\n🔬 STEP 5: Feature Engineering (TF-IDF)...")

"""
WHY TF-IDF over other methods:

1. vs Bag of Words: TF-IDF downweights common words ("the", "is")
2. vs Word2Vec: Simpler, faster, more interpretable
3. vs BERT: No GPU needed, much faster, easier to debug

TF-IDF works great for sentiment because:
- Words like "excellent", "terrible" have high importance
- Common words like "product", "bought" get lower weights
"""

tfidf = TfidfVectorizer(
    max_features=5000,      # Keep top 5000 words (balance performance vs memory)
    min_df=2,               # Word must appear in at least 2 documents
    max_df=0.8,             # Ignore words in >80% of documents (too common)
    ngram_range=(1, 2),     # Use unigrams AND bigrams ("not good" is meaningful)
    stop_words='english'    # NOW we remove stopwords (but keep negations in bigrams)
)

print("Fitting TF-IDF on training data...")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"Feature matrix shape: {X_train_tfidf.shape}")
print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

# Show top features
feature_names = tfidf.get_feature_names_out()
print(f"\n📊 Sample features: {list(feature_names[:20])}")

# ============================================================================
# STEP 6: MODEL TRAINING
# ============================================================================
print("\n🤖 STEP 6: Training Models...")

"""
MODEL CHOICES:

1. LOGISTIC REGRESSION
   - Simple, interpretable
   - Fast to train
   - Works well for text classification
   - Can see feature importance (coefficients)
   
2. MULTINOMIAL NAIVE BAYES
   - Classic text classification algorithm
   - Assumes features are independent (reasonable for words)
   - Very fast, good baseline
   - Often performs well despite simplicity

WHY NOT:
- Random Forest: Overkill for text, slower
- Deep Learning: Need more data, harder to explain
- SVM: Slower, less interpretable
"""

# Model 1: Logistic Regression
print("\n1️⃣ Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr_model.fit(X_train_tfidf, y_train)
print("✅ Logistic Regression trained")

# Model 2: Naive Bayes
print("\n2️⃣ Training Naive Bayes...")
nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train_tfidf, y_train)
print("✅ Naive Bayes trained")

# ============================================================================
# STEP 7: EVALUATION
# ============================================================================
print("\n📊 STEP 7: Model Evaluation...")

def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"{model_name} Results")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}  ← Of predicted Positive, how many are correct?")
    print(f"Recall:    {recall:.4f}  ← Of actual Positive, how many did we find?")
    print(f"F1-Score:  {f1:.4f}  ← Harmonic mean of Precision & Recall")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted Neg  Predicted Pos")
    print(f"Actual Negative:      {cm[0][0]}            {cm[0][1]}")
    print(f"Actual Positive:      {cm[1][0]}            {cm[1][1]}")
    
    # Classification Report
    print(f"\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred,
        'probabilities': y_proba
    }

# Evaluate both models
lr_results = evaluate_model(lr_model, X_test_tfidf, y_test, "LOGISTIC REGRESSION")
nb_results = evaluate_model(nb_model, X_test_tfidf, y_test, "NAIVE BAYES")

# ============================================================================
# STEP 8: MODEL COMPARISON
# ============================================================================
print("\n🏆 STEP 8: Model Comparison...")

comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Logistic Regression': [lr_results['accuracy'], lr_results['precision'], 
                            lr_results['recall'], lr_results['f1']],
    'Naive Bayes': [nb_results['accuracy'], nb_results['precision'], 
                    nb_results['recall'], nb_results['f1']]
})

print("\n" + comparison.to_string(index=False))

# Winner
if lr_results['f1'] > nb_results['f1']:
    winner = "Logistic Regression"
    best_model = lr_model
    print(f"\n🥇 WINNER: Logistic Regression (F1: {lr_results['f1']:.4f})")
else:
    winner = "Naive Bayes"
    best_model = nb_model
    print(f"\n🥇 WINNER: Naive Bayes (F1: {nb_results['f1']:.4f})")

# ============================================================================
# STEP 9: FEATURE IMPORTANCE (Logistic Regression)
# ============================================================================
print("\n🔍 STEP 9: Feature Importance Analysis...")

# Get top positive and negative features
lr_coef = lr_model.coef_[0]
top_positive_idx = lr_coef.argsort()[-10:][::-1]
top_negative_idx = lr_coef.argsort()[:10]

print("\n📈 Top 10 POSITIVE indicators:")
for idx in top_positive_idx:
    print(f"  '{feature_names[idx]}' → {lr_coef[idx]:.4f}")

print("\n📉 Top 10 NEGATIVE indicators:")
for idx in top_negative_idx:
    print(f"  '{feature_names[idx]}' → {lr_coef[idx]:.4f}")

# ============================================================================
# STEP 10: SAVE MODELS
# ============================================================================
print("\n💾 STEP 10: Saving Models...")

# Save vectorizer and best model
with open('/home/claude/sentiment-analysis/models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('/home/claude/sentiment-analysis/models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save model info
model_info = {
    'model_type': winner,
    'accuracy': lr_results['accuracy'] if winner == "Logistic Regression" else nb_results['accuracy'],
    'f1_score': lr_results['f1'] if winner == "Logistic Regression" else nb_results['f1'],
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'vocab_size': len(tfidf.vocabulary_)
}

with open('/home/claude/sentiment-analysis/models/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print("✅ Saved: tfidf_vectorizer.pkl")
print("✅ Saved: best_model.pkl")
print("✅ Saved: model_info.pkl")

# ============================================================================
# STEP 11: TEST ON NEW EXAMPLES
# ============================================================================
print("\n🧪 STEP 11: Testing on New Examples...")

test_reviews = [
    "This product is absolutely amazing! Best purchase ever.",
    "Terrible quality, waste of money. Very disappointed.",
    "Good product but delivery was slow.",
    "Love it! Highly recommend to everyone.",
    "Worst experience ever. Do not buy this."
]

print("\n" + "="*60)
for review in test_reviews:
    cleaned = preprocess_text(review)
    vectorized = tfidf.transform([cleaned])
    prediction = best_model.predict(vectorized)[0]
    proba = best_model.predict_proba(vectorized)[0]
    
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    confidence = proba[1] if prediction == 1 else proba[0]
    
    print(f"\nReview: {review}")
    print(f"→ {sentiment} (confidence: {confidence:.2%})")

print("\n" + "="*70)
print("✅ PIPELINE COMPLETE!")
print("="*70)
print(f"\n📈 Final Model: {winner}")
print(f"📊 Test Accuracy: {model_info['accuracy']:.4f}")
print(f"📊 Test F1-Score: {model_info['f1_score']:.4f}")
print(f"\n💡 Ready for deployment!")
