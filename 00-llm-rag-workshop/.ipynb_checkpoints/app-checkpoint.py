# app.py

import streamlit as st
from qa_bot_streamlit import qa_bot

# Streamlit app title and description
st.title("Course FAQ Bot")
st.write("Ask any question about course details and get instant answers!")

# User input for the question
user_question = st.text_input("Enter your question:")

if user_question:
    # Call QA bot function to get the answer
    answer = qa_bot(user_question)

    # Display the answer
    st.write(f"Answer: {answer}")
