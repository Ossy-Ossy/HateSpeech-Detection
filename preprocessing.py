import os
import re
import numpy as np
import emoji
import spacy
import torch
import shutil
import joblib
import streamlit as st
from transformers import AutoTokenizer, AutoModel

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_lg")

nlp = load_spacy()

tokenizer_zip = "tokenizer_sentiment_analysis.zip"
bert_model_zip = "bert_model_sentiment_analysis-20251106T011939Z-1-001.zip"
model_path = "model_sentiment_analysis.joblib" 
tokenizer_dir = "tokenizer"
bert_dir = "bert_model" 

for zip_file, target_dir in [
    ("tokenizer_sentiment_analysis.zip", "tokenizer"),
    ("bert_model_sentiment_analysis-20251106T011939Z-1-001.zip", "bert_model")
]:
    if os.path.exists(zip_file) and not os.path.exists(target_dir):
        shutil.unpack_archive(zip_file, target_dir)
        # Flatten if only one folder inside
        inner = os.path.join(target_dir, os.listdir(target_dir)[0])
        if os.path.isdir(inner):
            for f in os.listdir(inner):
                shutil.move(os.path.join(inner, f), target_dir)
            os.rmdir(inner)

      
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    bert_model = AutoModel.from_pretrained(bert_dir)
    model = joblib.load(model_path)
    bert_model.eval()
    return tokenizer, bert_model, model


def preprocess_text(text):
    text = str(text)
    emojis = ''.join(ch for ch in text if ch in emoji.EMOJI_DATA)
    text = re.sub(r'[^a-zA-Z\s]',' ' ,text)
    doc = nlp(text.lower())
    tokens = []
    
    for token in doc:
        if token.is_stop and token.lemma_ not in ['no' ,'never','not']:
            continue
        if not token.is_alpha:
            continue
        tokens.append(token.lemma_)
        
    if emojis:
        tokens.extend(list(emojis))
        
    return ' '.join(tokens)

def get_embeddings(text:str , tokenizer, bert_model , max_len: int = 128):
    processed = preprocess_text(text)
    
    inputs = tokenizer(
        processed,
        return_tensors="pt",        # Return PyTorch tensors
        truncation=True,
        padding="max_length",
        max_length=max_len
    )
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        # Use mean pooling to get a single vector
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def predict_sentiment(text):
    tokenizer, bert_model, model = load_models()
    embedding = get_embeddings(text ,tokenizer, bert_model)
    embedding = embedding.reshape(1,-1)
    prediction = model.predict(embedding)[0]
    if prediction == 0:
        return "No Hate Speech Detected"
    else:
        return "Hate Speech Detected"

