# qa_bot-streamlit.py

import json
import requests
from elasticsearch import Elasticsearch, exceptions as es_exceptions
from tqdm.auto import tqdm
from openai import OpenAI, APIError
import streamlit as st

# Constants
DOCUMENTS_URL = "https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json"
INDEX_NAME = "course-questions"

def load_and_process_documents(url):
    """
    Load and process documents from a JSON file hosted at the given URL.
    """
    # Download JSON file using requests
    response = requests.get(url)
    if response.status_code == 200:
        # Save JSON content to a local file
        with open('documents.json', 'wb') as f_out:
            f_out.write(response.content)
        
        # Load JSON file
        with open('./documents.json', 'rt') as f_in:
            documents_file = json.load(f_in)

        # Process documents
        documents = []
        doc_id = 1  # Initialize document ID counter
        for course in documents_file:
            course_name = course['course']
            for doc in course['documents']:
                doc['course'] = course_name
                doc['id'] = doc_id  # Assign ID to each document
                documents.append(doc)
                doc_id += 1  # Increment document ID counter
        
        return documents
    else:
        st.error(f"Failed to download documents from {url}. Status code: {response.status_code}")
        return []

def create_elasticsearch_index(es, index_name):
    """
    Create an Elasticsearch index with specified settings and mappings.
    """
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"}
            }
        }
    }

    try:
        # Attempt to create the index
        response = es.indices.create(index=index_name, body=index_settings)
        st.success(f"Index '{index_name}' created successfully: {response}")
    except es_exceptions.RequestError as e:
        # Handle index creation errors
        if e.error == 'resource_already_exists_exception':
            st.warning(f"Index '{index_name}' already exists.")
        else:
            st.error(f"Failed to create index '{index_name}': {e}")

def index_documents(es, index_name, documents):
    """
    Index documents into the specified Elasticsearch index.
    """
    for doc in tqdm(documents, desc="Indexing documents"):
        es.index(index=index_name, document=doc)

def retrieve_documents(es, index_name, user_question, course_name="data-engineering-zoomcamp", max_results=5):
    """
    Retrieve relevant documents from Elasticsearch based on the user question and course.
    """
    search_query = {
        "size": max_results,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": user_question,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": course_name
                    }
                }
            }
        }
    }

    # Execute search query
    response = es.search(index=index_name, body=search_query)
    documents = [hit['_source'] for hit in response['hits']['hits']]
    return documents

def build_context(documents):
    """
    Build a formatted context string based on retrieved documents.
    """
    # Template for formatting document information in the context
    context_template = """
    Section: {section}
    Question: {question}
    Answer: {text}
    """.strip()

    context_result = ""
    for doc in documents:
        doc_str = context_template.format(**doc)
        context_result += ("\n\n" + doc_str)
    
    return context_result.strip()

def build_prompt(user_question, context):
    """
    Build a prompt string based on the user question and retrieved documents.
    """
    # Template for the prompt presented to the OpenAI model
    prompt_template = """
    You're a course teaching assistant.
    Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database.
    Don't use other information outside of the provided CONTEXT.  

    QUESTION: {user_question}

    CONTEXT:

    {context}
    """.strip()

    prompt = prompt_template.format(
        user_question=user_question,
        context=context
    )
    return prompt

def ask_openai(prompt, client, fallback_model="gpt-3.5-turbo"):
    """
    Send a prompt to the OpenAI chat model and retrieve the model's response.
    """
    try:
        # Attempt to use the primary model (gpt-4o)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
    except APIError as e:
        # Handle error if primary model is not found or accessible
        if e.status_code == 404 and 'model_not_found' in str(e):
            st.warning("GPT-4o model not found or access denied. Falling back to GPT-3.5-turbo...")
            # Fallback to secondary model (gpt-3.5-turbo)
            response = client.chat.completions.create(
                model=fallback_model,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content
        else:
            # Handle other API errors
            st.error(f"Error with GPT-4o model: {e}")
            answer = "NONE"  # Provide a default answer or indication of failure

    return answer

def qa_bot(user_question):
    """
    Answer a user question by querying the FAQ database, constructing a prompt, 
    and using an OpenAI model to generate the answer.
    """
    # Initialize Elasticsearch client
    es = Elasticsearch("http://localhost:9200")

    # Load and process documents from JSON file
    documents = load_and_process_documents(DOCUMENTS_URL)

    # Create Elasticsearch index if not exists
    create_elasticsearch_index(es, INDEX_NAME)

    # Index documents into Elasticsearch
    index_documents(es, INDEX_NAME, documents)

    # Retrieve relevant documents based on user question
    context_docs = retrieve_documents(es, INDEX_NAME, user_question)

    # Build prompt for OpenAI model
    context = build_context(context_docs)
    prompt = build_prompt(user_question, context)

    # Initialize OpenAI client
    client = OpenAI()

    # Ask OpenAI model for an answer based on the prompt
    answer = ask_openai(prompt, client)

    return answer

# Entry point of the script
if __name__ == "__main__":
    # Example usage: specify a user question and get an answer
    user_question = "How do I join the course after it has started?"
    answer = qa_bot(user_question)
    st.write(f"Answer: {answer}")
