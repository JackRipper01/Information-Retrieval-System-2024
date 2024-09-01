import numpy as np
import streamlit as st
from document_processor import DocumentProcessor
from custom_bm25 import BM25

# Path to your dataset
file_path = r"datasets\cranfield_dataset.txt"


@st.cache_data
def load_documents(file_path):
    print("Processing documents...")
    document_processor = DocumentProcessor(file_path)
    return document_processor


@st.cache_resource
def initialize_bm25(documents):
    print("Calculating BM25 and Cosine Similarity scores...")
    return BM25(documents)


# Load and process documents
document_processor = load_documents(file_path)
documents = document_processor.documents

# Search for the word "investigations" in the content of each document
search_word = "investigations"
for doc in documents:
    if search_word in doc['content']:
        print(f"Word '{search_word}' found in document ID: {doc['id']}")
        break

# Instantiate the BM25 class with the processed documents
bm25 = initialize_bm25(documents)

# Streamlit app
st.title("Information Retrieval System ")

# User input for search query
query = st.text_input("Enter your search here:")

if query:

    # Display results
    st.markdown("<h2 style='text-align: center;'>Results</h2>",
                unsafe_allow_html=True)
    # Calculate combined scores for the query
    tuple_bm25_cos_doc = bm25.get_bm25_combined_cosine_sim_scores(query)

    # Use a list comprehension instead of a set comprehension
    output = [(bm25_score, cosine_score, document_processor.raw_documents[int(
        doc['id']) - 1]) for bm25_score, cosine_score, doc in tuple_bm25_cos_doc]

    for bm25_score, cosine_score, doc in output:
        st.write(
            f"<div style='font-size: 24px; margin-bottom: 10px;'><b>Title:</b> {doc['title']}</div>"
            f"<div style='font-size: 16px; margin-bottom: 10px;'><b>BM25 Score:</b> {bm25_score:.4f}</div>"
            f"<div style='font-size: 16px; margin-bottom: 10px;'><b>Cosine Similarity Score:</b> {cosine_score:.4f}</div>"
            f"<div style='font-size: 16px; margin-bottom: 10px;'><b>Document ID:</b> {doc['id']}</div>"
            f"<div style='font-size: 16px; margin-bottom: 10px;'><b>Author:</b> {doc['author']}</div>"
            f"<div style='font-size: 16px; margin-bottom: 10px;'><b>Bibliography:</b> {doc['bibliography']}</div>"
            f"<div style='font-size: 24px; margin-bottom: 10px;'><b>Content:</b> {doc['content']}</div>"
            "<hr>",
            "<hr>",
            "<hr>",
            "<hr>",
            unsafe_allow_html=True
        )
