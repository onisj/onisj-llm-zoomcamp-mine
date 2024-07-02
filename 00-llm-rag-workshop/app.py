import streamlit as st
from qa_bot_streamlit import qa_bot

# Streamlit app title and description
st.title("Course FAQ Bot")

st.markdown("""
<details>
  <summary>About This FAQ Bot</summary>
  This is a FAQ bot that uses Large Language Models (LLMs) to provide instant answers 
  based on course-related questions. It retrieves information from a database of course 
  documents and uses an OpenAI model to generate responses. The bot aims to assist users 
  with queries about course details efficiently. 
  <br><br>
  For more details and complete code, check out the <a href="https://github.com/onisj/onisj-llm-zoomcamp-mine/tree/main/00-llm-rag-workshop" target="_blank">GitHub repository</a>.
</details>
""", unsafe_allow_html=True)

# Adding a line break
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


st.write("Ask any question about course details and get instant answers!")

# User input for the question
user_question = st.text_input("Enter your question:")

if user_question:
    # Call QA bot function to get the answer
    answer = qa_bot(user_question)

    # Display the answer
    st.write(f"Answer: {answer}")
