import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util

# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AAU AI Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AAU AI Chat Assistant (FAISS + SBERT)")
st.caption("Semantic FAQ System for Addis Ababa University")

# =========================================================
# 2. LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2",
        cache_folder="./models"
    )

model = load_model()

# =========================================================
# 3. LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    with open("faq_data.json", "r", encoding="utf-8") as f:
        return json.load(f)

faq_data = load_data()

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# =========================================================
# 4. PREPROCESS + QUERY EXPANSION
# =========================================================
def preprocess(text):
    return text.lower().strip()

def expand_query(query):
    synonyms = {
        "admission": "apply entry requirement university",
        "apply": "application admission requirement",
        "fee": "tuition cost payment price",
        "deadline": "due date submission",
        "exam": "test entrance assessment"
    }

    for key, value in synonyms.items():
        if key in query:
            query += " " + value

    return query

# =========================================================
# 5. EMBEDDINGS
# =========================================================
@st.cache_resource
def build_embeddings():
    return model.encode(questions, normalize_embeddings=True)

embeddings = build_embeddings()

# =========================================================
# 6. FAISS INDEX
# =========================================================
@st.cache_resource
def build_index():
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))
    return index

index = build_index()

# =========================================================
# 7. SEARCH FUNCTION
# =========================================================
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

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results

# =========================================================
# 8. CONFIDENCE CHECK
# =========================================================
def is_confident(results):
    return len(results) > 0 and results[0]["score"] > 0.35

# =========================================================
# 9. TEST DATASET (EVALUATION)
# =========================================================
test_cases = [
    {
        "question": "How much do I need to pay for graduate application?",
        "expected": "The application fee for Graduate Speciality programs is ETB 500."
    },
    {
        "question": "Where should I submit my documents?",
        "expected": "AAU Admission Office, Main Campus Registrar Building, Room 203."
    },
    {
        "question": "Can I reuse my admission test result?",
        "expected": "No. The Undergraduate Admission Test (UAT) result is valid only for the specific application cycle."
    },
    {
        "question": "What is the tuition fee payment method?",
        "expected": "Tuition fees are usually paid in installments per semester."
    }
]

# =========================================================
# 10. EVALUATION METRICS
# =========================================================
def evaluate_system():
    correct_top1 = 0
    recall_at_k = 0
    mrr_total = 0

    for item in test_cases:
        results = search(item["question"], top_k=3)

        # Top-1 Accuracy
        if item["expected"] in results[0]["answer"]:
            correct_top1 += 1

        # Recall@3
        if any(item["expected"] in r["answer"] for r in results):
            recall_at_k += 1

        # MRR
        for rank, r in enumerate(results, start=1):
            if item["expected"] in r["answer"]:
                mrr_total += 1 / rank
                break

    n = len(test_cases)

    return {
        "Top-1 Accuracy": correct_top1 / n,
        "Recall@3": recall_at_k / n,
        "MRR": mrr_total / n
    }

# =========================================================
# 11. SIDEBAR EVALUATION DASHBOARD
# =========================================================
st.sidebar.title("📊 Evaluation Dashboard")

if st.sidebar.button("Run Evaluation"):
    metrics = evaluate_system()

    st.sidebar.success("Evaluation Completed")

    st.sidebar.write(f"Top-1 Accuracy: {metrics['Top-1 Accuracy']:.2f}")
    st.sidebar.write(f"Recall@3: {metrics['Recall@3']:.2f}")
    st.sidebar.write(f"MRR: {metrics['MRR']:.2f}")

# =========================================================
# 12. CHAT MEMORY
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# 13. DISPLAY CHAT HISTORY
# =========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================================================
# 14. USER INPUT (CHAT UI)
# =========================================================
user_input = st.chat_input("Ask anything about AAU...")

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    results = search(user_input)
    best = results[0]

    if not is_confident(results):
        response = (
            "❌ I couldn't find a confident answer.\n"
            "Please rephrase your question."
        )
    else:
        context = "\n\n".join(
            f"Q: {r['question']}\nA: {r['answer']}"
            for r in results[:3]
        )

        response = f"""
✅ **Answer:**
{best['answer']}

---

🔎 **Related FAQs:**
{context}
"""

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.markdown(response)

        with st.expander("🔍 Top Matches (Debug)"):
            for r in results:
                st.markdown(f"**Q:** {r['question']}")
                st.markdown(f"**Score:** {r['score']:.3f}")
                st.markdown("---")

# =========================================================
# 15. DATASET VIEWER
# =========================================================
with st.expander("📚 View FAQ Dataset"):
    st.json(faq_data)