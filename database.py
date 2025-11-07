import sqlite3
from datetime import datetime
import streamlit as st 
import joblib

def create_db():
    conn = sqlite3.connect('sentiment.db')
    c = conn.cursor()
    c.execute(
        '''
        CREATE TABLE IF NOT EXISTS sentiment_analysis(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )
    conn.commit()
    conn.close()
    
def insert_data(text, sentiment):
    conn = sqlite3.connect('sentiment.db')
    c = conn.cursor()
    c.execute(
        'INSERT INTO sentiment_analysis (text, sentiment) VALUES (?, ?)', (text, sentiment)
    )
    conn.commit()
    conn.close()
    
def fetch_all():
    conn = sqlite3.connect('sentiment.db')
    c = conn.execute('SELECT * FROM sentiment_analysis ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    return rows

