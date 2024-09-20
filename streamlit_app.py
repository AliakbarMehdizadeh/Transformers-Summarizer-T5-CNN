import streamlit as st
import requests

# FastAPI URL
API_URL = "http://localhost:8000/summarize/"

st.title("Text Summarization")

# Text input
text = st.text_area("Enter the document to summarize", height=300)

if st.button("Summarize"):
    # Send POST request to the API
    response = requests.post(API_URL, json={"text": text})
    
    # Display the summarized text
    if response.status_code == 200:
        summary = response.json()["summary"]
        st.write("**Summary:**")
        st.write(summary)
    else:
        st.write("Error: Failed to generate summary.")
