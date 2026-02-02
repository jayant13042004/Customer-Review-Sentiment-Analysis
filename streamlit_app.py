"""
CUSTOMER REVIEW SENTIMENT ANALYZER
Streamlit Web App for Production Deployment

Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pickle
import re
import string
import pandas as pd
import numpy as np
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .positive-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .negative-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-low {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load trained models and vectorizer"""
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    
    return vectorizer, model, model_info

def preprocess_text(text):
    """Clean text for prediction"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

def predict_sentiment(text, vectorizer, model):
    """
    Predict sentiment for given text
    
    Returns:
        sentiment: 'Positive' or 'Negative'
        confidence: float (0-1)
        probabilities: [neg_prob, pos_prob]
    """
    # Preprocess
    cleaned = preprocess_text(text)
    
    # Vectorize
    vectorized = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probabilities[1] if prediction == 1 else probabilities[0]
    
    return sentiment, confidence, probabilities

# Load models
try:
    vectorizer, model, model_info = load_models()
    models_loaded = True
except:
    models_loaded = False
    st.error("⚠️ Models not found! Please train models first by running: python notebooks/train_model.py")

# Header
st.markdown('<h1 class="main-header">🛍️ Customer Review Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Sentiment Analysis for Retail")

if models_loaded:
    # Sidebar - Model Info
    st.sidebar.header("📊 Model Information")
    st.sidebar.metric("Model Type", model_info['model_type'])
    st.sidebar.metric("Test Accuracy", f"{model_info['accuracy']:.2%}")
    st.sidebar.metric("F1-Score", f"{model_info['f1_score']:.4f}")
    st.sidebar.metric("Training Samples", f"{model_info['train_samples']:,}")
    
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ How It Works")
    st.sidebar.info("""
    1. **Input**: Customer review text
    2. **Preprocessing**: Clean and normalize
    3. **Vectorization**: Convert to TF-IDF features
    4. **Prediction**: ML model classifies sentiment
    5. **Output**: Sentiment + confidence score
    """)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze Review", "📊 Batch Analysis", "ℹ️ About"])
    
    # TAB 1: Single Review Analysis
    with tab1:
        st.header("Analyze Single Review")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            review_text = st.text_area(
                "Enter customer review:",
                height=150,
                placeholder="Example: This product is amazing! Fast delivery and great quality."
            )
        
        with col2:
            st.markdown("### Quick Examples")
            if st.button("😊 Positive Example"):
                review_text = "Excellent product! Highly recommended. Fast delivery and great packaging."
                st.session_state['example_text'] = review_text
            
            if st.button("😞 Negative Example"):
                review_text = "Terrible quality. Waste of money. Very disappointed with this purchase."
                st.session_state['example_text'] = review_text
            
            if st.button("🤔 Mixed Example"):
                review_text = "Product is okay but delivery was very late. Not as described in pictures."
                st.session_state['example_text'] = review_text
        
        # Use example if button clicked
        if 'example_text' in st.session_state:
            review_text = st.session_state['example_text']
            st.session_state.pop('example_text')
        
        if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
            if review_text.strip():
                with st.spinner("Analyzing..."):
                    sentiment, confidence, probabilities = predict_sentiment(review_text, vectorizer, model)
                
                # Display results
                if sentiment == "Positive":
                    st.markdown(f"""
                    <div class="positive-box">
                        <h2>✅ POSITIVE REVIEW</h2>
                        <p style="font-size: 1.2rem;">Confidence: <span class="{'confidence-high' if confidence > 0.8 else 'confidence-low'}">{confidence:.1%}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="negative-box">
                        <h2>❌ NEGATIVE REVIEW</h2>
                        <p style="font-size: 1.2rem;">Confidence: <span class="{'confidence-high' if confidence > 0.8 else 'confidence-low'}">{confidence:.1%}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.subheader("📊 Probability Breakdown")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Negative Probability", f"{probabilities[0]:.2%}")
                with col2:
                    st.metric("Positive Probability", f"{probabilities[1]:.2%}")
                
                # Confidence interpretation
                st.subheader("💡 Confidence Interpretation")
                if confidence >= 0.9:
                    st.success("🟢 **High Confidence** - Model is very sure about this prediction")
                elif confidence >= 0.7:
                    st.info("🟡 **Medium Confidence** - Model is reasonably confident")
                else:
                    st.warning("🟠 **Low Confidence** - Review might have mixed sentiment or be ambiguous")
                
                # Action recommendation
                st.subheader("🎯 Recommended Action")
                if sentiment == "Negative" and confidence > 0.7:
                    st.error("**URGENT:** Contact customer immediately to resolve issue")
                elif sentiment == "Negative":
                    st.warning("**FOLLOW-UP:** Review complaint and plan response")
                elif sentiment == "Positive" and confidence > 0.9:
                    st.success("**CELEBRATE:** Share positive feedback with team")
                else:
                    st.info("**MONITOR:** Keep track for trends")
            
            else:
                st.warning("⚠️ Please enter a review text")
    
    # TAB 2: Batch Analysis
    with tab2:
        st.header("Batch Review Analysis")
        st.markdown("Upload a CSV file with customer reviews for bulk analysis")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("### Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Select column
            text_column = st.selectbox("Select review text column:", df.columns)
            
            if st.button("📊 Analyze All Reviews"):
                with st.spinner("Analyzing all reviews..."):
                    # Predict all
                    sentiments = []
                    confidences = []
                    
                    for text in df[text_column]:
                        sentiment, confidence, _ = predict_sentiment(str(text), vectorizer, model)
                        sentiments.append(sentiment)
                        confidences.append(confidence)
                    
                    df['Predicted_Sentiment'] = sentiments
                    df['Confidence'] = [f"{c:.2%}" for c in confidences]
                
                # Show results
                st.success(f"✅ Analyzed {len(df)} reviews!")
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_positive = (df['Predicted_Sentiment'] == 'Positive').sum()
                    st.metric("Positive Reviews", f"{n_positive} ({n_positive/len(df)*100:.1f}%)")
                with col2:
                    n_negative = (df['Predicted_Sentiment'] == 'Negative').sum()
                    st.metric("Negative Reviews", f"{n_negative} ({n_negative/len(df)*100:.1f}%)")
                with col3:
                    avg_conf = np.mean(confidences)
                    st.metric("Avg Confidence", f"{avg_conf:.1%}")
                
                # Show results
                st.write("### Analysis Results:")
                st.dataframe(df)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "sentiment_analysis_results.csv",
                    "text/csv"
                )
    
    # TAB 3: About
    with tab3:
        st.header("ℹ️ About This Project")
        
        st.markdown("""
        ## Customer Review Sentiment Analysis
        
        ### 🎯 Business Problem
        This system automatically analyzes customer reviews to identify satisfaction levels.
        
        **Key Benefits:**
        - ⚡ **Speed**: Analyze thousands of reviews in seconds
        - 🎯 **Accuracy**: {:.1%} accuracy on test data
        - 💰 **Cost Savings**: Reduce manual review processing time by 95%
        - 📈 **Insights**: Track sentiment trends over time
        
        ### 🤖 How It Works
        
        1. **Text Preprocessing**
           - Clean and normalize review text
           - Remove URLs, punctuation, extra spaces
           
        2. **Feature Extraction (TF-IDF)**
           - Convert text to numerical features
           - Weight important words higher
           
        3. **Classification ({})**
           - Trained on thousands of real reviews
           - Predicts: Positive or Negative
           
        4. **Confidence Score**
           - Shows model certainty (0-100%)
           - Helps prioritize manual review
        
        ### 📊 Model Performance
        
        - **Accuracy**: {:.2%}
        - **F1-Score**: {:.4f}
        - **Training Data**: {:,} reviews
        - **Test Data**: {:,} reviews
        
        ### 🏢 Business Applications
        
        **For Reliance Retail:**
        - Flag negative reviews for immediate customer service response
        - Identify problematic product categories
        - Monitor brand sentiment over time
        - Prioritize high-value customer issues
        - Generate automated sentiment reports
        
        ### 🚀 Production Deployment
        
        **Current**: Local Streamlit app
        
        **Production Options**:
        - Deploy on AWS/GCP/Azure
        - Integrate with review API
        - Real-time sentiment dashboard
        - Automated alert system
        
        ### 👨‍💻 Technical Stack
        
        - **ML**: scikit-learn, TF-IDF
        - **Model**: {}
        - **Frontend**: Streamlit
        - **Language**: Python 3.x
        
        ### 📈 Future Improvements
        
        - Multi-class sentiment (Very Positive, Positive, Neutral, Negative, Very Negative)
        - Aspect-based sentiment (e.g., "good product but slow delivery")
        - Multilingual support
        - Real-time model updates
        - Integration with CRM systems
        
        ### 📞 Contact
        
        Built for ML Engineer interview at Reliance Retail.
        """.format(
            model_info['accuracy'],
            model_info['model_type'],
            model_info['accuracy'],
            model_info['f1_score'],
            model_info['train_samples'],
            model_info['test_samples'],
            model_info['model_type']
        ))

else:
    st.error("Models not loaded. Please train models first!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    f"Powered by {model_info['model_type'] if models_loaded else 'ML'} | "
    f"Accuracy: {model_info['accuracy']:.1%} | " if models_loaded else ""
    "Built with ❤️ for Reliance Retail"
    "</div>",
    unsafe_allow_html=True
)
