import streamlit as st

from preprocessing import predict_sentiment
from database import create_db ,insert_data ,fetch_all

st.sidebar.title("ℹ️ About this App")
st.sidebar.markdown("""
This **Hate Speech Detection App** uses **Machine Learning and Natural Language Processing (NLP)** techniques 
to analyze text and determine whether it contains **hate speech** or not.

### How it works:
1. The text is cleaned and tokenized using **spaCy** and **emoji** handling.
2. A **BERT model** generates contextual embeddings.
3. A trained **Neural Network classifier** (stored as a `.joblib` file) predicts if the text is **Hate Speech** or **Not Hate Speech**.
4. All predictions are stored in a **SQLite database** for tracking.

🧠 Built with: `Streamlit`, `PyTorch`, `Transformers`, `spaCy`, and `Joblib`
""")

st.title("🧠 Hate Speech Detection App")
st.write("Enter any text below to check whether it contains hate speech.")

create_db()

user_text = st.text_area('Paste your text here:')

if st.button('Analyze sentiment'):
    if not user_text.strip():
        st.warning("Please enter some text")
        
    else:
        sentiment = predict_sentiment(user_text)
        st.success(f"Sentiment: **{sentiment}**")
        insert_data(user_text, sentiment)
        
if st.checkbox("📜 Show Past Entries"):
    rows = fetch_all()
    if rows:
        for row in rows:
            st.write(f"🕓 {row[3]} | **Text:** {row[1]} → **Sentiment:** {row[2]}")
    else:
        st.info("No entries yet.")
