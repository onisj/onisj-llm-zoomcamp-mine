import streamlit as st
import pandas as pd
import numpy as np
import requests
import torch
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# CourseFAQBot class
class CourseFAQBot:
    def __init__(self, model_name="bert-base-uncased", docs_url=None, batch_size=8):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode if not training
        self.batch_size = batch_size
        self.df = self._download_and_process_documents(docs_url)
        self.document_embeddings = self.compute_embeddings(self.df['text'].tolist())

    def _download_and_process_documents(self, docs_url):
        """
        Download and process the document data.
        """
        docs_response = requests.get(docs_url)
        documents_raw = docs_response.json()
        
        documents = []
        for course in documents_raw:
            course_name = course['course']
            for doc in course['documents']:
                doc['course'] = course_name
                documents.append(doc)
        
        # Create the DataFrame
        return pd.DataFrame(documents, columns=['course', 'section', 'question', 'text'])

    def make_batches(self, seq, n):
        """
        Split a sequence into batches of size n.
        """
        result = []
        for i in range(0, len(seq), n):
            batch = seq[i:i+n]
            result.append(batch)
        return result

    def compute_embeddings(self, texts):
        """
        Compute embeddings for a list of texts using a pre-trained transformer model.
        """
        text_batches = self.make_batches(texts, self.batch_size)
        all_embeddings = []
        
        for batch in tqdm(text_batches, desc="Computing embeddings"):
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**encoded_input)
                hidden_states = outputs.last_hidden_state
                batch_embeddings = hidden_states.mean(dim=1)
                batch_embeddings_np = batch_embeddings.cpu().numpy()
                all_embeddings.append(batch_embeddings_np)
        
        final_embeddings = np.vstack(all_embeddings)
        return final_embeddings

    def query(self, query_text, top_n=10):
        """
        Perform a query to find the most relevant documents.
        """
        query_embedding = self.compute_embeddings([query_text])
        similarities = cosine_similarity(query_embedding, self.document_embeddings).flatten()
        top_n_indices = similarities.argsort()[-top_n:][::-1]
        top_n_documents = self.df.iloc[top_n_indices]
        return top_n_documents

# Streamlit application
st.title("FAQ Search Engine for DataTalks")

# Initialize CourseFAQBot
docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
faq_bot = CourseFAQBot(docs_url=docs_url)

# Input fields for query and filters
query = st.text_input("Enter your query:")
courses = st.multiselect("Select course(s):", options=faq_bot.df['course'].unique())

# Search button
if st.button("Search"):
    results = faq_bot.query(query)
    
    # Filter results by selected courses if any
    if courses:
        results = results[results['course'].isin(courses)]
    
    # Display results with space in between
    for i, result in enumerate(results.to_dict(orient='records')):
        st.write(f"### Result {i+1}")
        st.write(f"**Course**: {result['course']}")
        st.write(f"**Section**: {result['section']}")
        st.write(f"**Question**: {result['question']}")
        st.write(f"**Text**: {result['text']}")
        st.write("")  # Adds a blank space between results
        st.markdown("---")
