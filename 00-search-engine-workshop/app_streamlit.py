# Import necessary libraries

import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to fetch data
def fetch_data():
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()
    documents = []

    for course in documents_raw:
        course_name = course['course']
        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)
    
    return pd.DataFrame(documents, columns=['course', 'section', 'question', 'text'])

# TextSearch class
class TextSearch:
    def __init__(self, text_fields):
        self.text_fields = text_fields
        self.matrices = {}
        self.vectorizers = {}

    def fit(self, records, vectorizer_params={}):
        self.df = pd.DataFrame(records)
        for f in self.text_fields:
            cv = TfidfVectorizer(**vectorizer_params)
            X = cv.fit_transform(self.df[f])
            self.vectorizers[f] = cv
            self.matrices[f] = X

    def search(self, query, filters={}, boost={}):
        score = np.zeros(len(self.df))
        for f in self.text_fields:
            b = boost.get(f, 1.0)
            q = self.vectorizers[f].transform([query])
            s = cosine_similarity(self.matrices[f], q).flatten()
            score = score + b * s
        
        for field, value in filters.items():
            mask = (self.df[field] == value).values
            score = score * mask
        
        idx = np.argsort(-score)[:5]
        return self.df.iloc[idx].to_dict(orient='records')

# Main Streamlit application
st.title("FAQ Search Engine for DataTalks")

# Load data
df = fetch_data()

# Initialize TextSearch
text_search = TextSearch(text_fields=['section', 'question', 'text'])
text_search.fit(df.to_dict(orient='records'), vectorizer_params={'stop_words': 'english', 'min_df': 3})

# Input fields for query and filters
query = st.text_input("Enter your query:")
courses = st.multiselect("Select course(s): Pls, select just one", options=df['course'].unique())

# Search button
if st.button("Search"):
    filters = {}
    if courses:
        filters['course'] = courses[0] if len(courses) == 1 else courses
    results = text_search.search(query, filters=filters, boost={'question': 3.0})

    for i, result in enumerate(results):
        st.write(f"### Result {i+1}")
        st.write(f"**Course**: {result['course']}")
        st.write(f"**Section**: {result['section']}")
        st.write(f"**Question**: {result['question']}")
        st.write(f"**Text**: {result['text']}")
        st.write("")  
        st.markdown("---")