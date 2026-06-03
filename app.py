import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AAU AI Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AAU AI Chat Assistant (FAISS + Semantic Search)")
st.caption("Improved Semantic FAQ system using Sentence-BERT + FAISS (No LLM layer)")

# -----------------------------
# 2. LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2",
        cache_folder="./models"
    )

model = load_model()

# -----------------------------
# 3. LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    with open("faq_data.json", "r", encoding="utf-8") as f:
        return json.load(f)

faq_data = load_data()

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# -----------------------------
# 4. QUERY PREPROCESSING
# -----------------------------
def preprocess(text):
    return text.lower().strip()

# -----------------------------
# 5. QUERY EXPANSION (Simple NLP Trick)
# -----------------------------
def expand_query(query):
    synonyms = {
        "admission": "admission entry requirement apply university",
        "apply": "application admission entry requirement",
        "exam": "entrance exam requirement test",
        "fee": "payment tuition cost price",
        "deadline": "due date submission deadline"
    }

    for key, value in synonyms.items():
        if key in query:
            query += " " + value

    return query

# -----------------------------
# 6. EMBEDDINGS
# -----------------------------
@st.cache_resource
def build_embeddings():
    return model.encode(questions, normalize_embeddings=True)

embeddings = build_embeddings()

# -----------------------------
# 7. FAISS INDEX
# -----------------------------
@st.cache_resource
def build_index():
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))
    return index

index = build_index()

# -----------------------------
# 8. SEARCH FUNCTION (TOP-K + IMPROVED LOGIC)
# -----------------------------
def search(query, top_k=5):
    query = preprocess(query)
    query = expand_query(query)

    query_vec = model.encode([query], normalize_embeddings=True)

    scores, indices = index.search(np.array(query_vec), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "question": questions[idx],
            "answer": answers[idx],
            "score": float(score)
        })

    # Rerank (extra safety step)
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results

# -----------------------------
# 9. CONFIDENCE CHECK (IMPORTANT IMPROVEMENT)
# -----------------------------
def is_confident(results):
    if len(results) < 2:
        return False

    top_score = results[0]["score"]
    second_score = results[1]["score"]

    return top_score > 0.45 and (top_score - second_score) > 0.05

# -----------------------------
# 10. CHAT MEMORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# 11. DISPLAY HISTORY
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# 12. USER INPUT
# -----------------------------
user_input = st.chat_input("Ask anything about AAU...")

if user_input:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # -------------------------
    # SEARCH
    # -------------------------
    results = search(user_input)

    best = results[0]

    # -------------------------
    # RESPONSE LOGIC (IMPROVED)
    # -------------------------
    if not is_confident(results):
        response = (
            "❌ I couldn't find a highly confident answer.\n\n"
            "Try rephrasing your question or using different keywords."
        )
    else:
        # Use TOP-3 context (more intelligent behavior)
        context = "\n\n".join([
            f"Q: {r['question']}\nA: {r['answer']}"
            for r in results[:3]
        ])

        # ChatGPT-like structured response (WITHOUT LLM)
        response = f"""
✅ **Best Answer:**
{best['answer']}

---

🔎 **Related FAQ Context:**
{context}
"""

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(response)

        # Debug section
        with st.expander("🔍 Top Matching FAQs (Debug View)"):
            for r in results:
                st.markdown(f"**Q:** {r['question']}")
                st.markdown(f"**Score:** {r['score']:.3f}")
                st.markdown("---")