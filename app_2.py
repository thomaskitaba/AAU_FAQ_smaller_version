import streamlit as st
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parent / "faq_data.json"

# Load data
@st.cache_data
def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

faq_data = load_data()
model = load_model()

# Precompute embeddings once (cached)
@st.cache_resource
def get_embeddings():
    questions = [item["question"] for item in faq_data]
    embeddings = model.encode(questions)
    return questions, embeddings

questions, embeddings = get_embeddings()

# Streamlit UI
st.title("🎓 AAU FAQ Assistant")

query = st.text_input("Ask your question:")

if query:
    query_emb = model.encode([query])

    scores = cosine_similarity(query_emb, embeddings)[0]
    best_idx = np.argmax(scores)

    st.subheader("Answer")
    st.write(faq_data[best_idx]["answer"])

    st.subheader("Related Question")
    st.write(faq_data[best_idx]["question"])