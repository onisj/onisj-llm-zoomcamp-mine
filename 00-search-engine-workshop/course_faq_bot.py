# Install necessary libraries
# pip install transformers pandas tqdm numpy scikit-learn requests

import torch
from transformers import BertModel, BertTokenizer
import requests
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

    def query(self, query_text, course_name=None, top_n=5):
        """
        Perform a query to find the most relevant documents, optionally filtering by course.

        Args:
            query_text (str): The query text to search for.
            course_name (str, optional): The name of the course to filter by. Defaults to None.
            top_n (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            pd.DataFrame: A DataFrame containing the top N most relevant documents.
        """
        # Filter the DataFrame by course name if provided
        if course_name:
            df_filtered = self.df[self.df['course'] == course_name]
            document_embeddings = self.compute_embeddings(df_filtered['text'].tolist())
        else:
            df_filtered = self.df
            document_embeddings = self.document_embeddings
        
        # Compute embedding for the query
        query_embedding = self.compute_embeddings([query_text])
        similarities = cosine_similarity(query_embedding, document_embeddings).flatten()
        
        # Get the indices of the top N most similar documents
        top_n_indices = similarities.argsort()[-top_n:][::-1]
        
        # Retrieve the corresponding rows from the filtered DataFrame
        top_n_documents = df_filtered.iloc[top_n_indices]
        return top_n_documents

# Importing the document from the URL
docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
faq_bot = CourseFAQBot(docs_url=docs_url)

course = 'data-engineering-zoomcamp'

# Your query text
query_text = "Can I still join the course after the start date?"

# Get top 5 most relevant documents for the specified course
top_documents = faq_bot.query(query_text, course_name=course)
top_documents
