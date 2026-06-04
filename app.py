import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util

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
# 5. QUERY EXPANSION
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
# 8. SEARCH FUNCTION
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

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results

# -----------------------------
# 9. CONFIDENCE CHECK
# -----------------------------
def is_confident(results):
    if len(results) < 2:
        return False
    return results[0]["score"] > 0.35

# -----------------------------
# 10. CHAT MEMORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# 🧪 EVALUATION DATASET (NEW SECTION)
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
        "expected": "Tuition fees are usually paid in installments per semester, typically two installments per semester or four per year depending on the program."
    }
]

# =========================================================
# 📊 EVALUATION FUNCTIONS
# =========================================================
def semantic_score(a, b):
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


def run_evaluation():
    results_summary = []
    correct = 0

    for item in test_cases:
        pred_results = search(item["question"])
        prediction = pred_results[0]["answer"]

        score = semantic_score(prediction, item["expected"])

        is_correct = score >= 0.75
        if is_correct:
            correct += 1

        results_summary.append({
            "question": item["question"],
            "prediction": prediction,
            "expected": item["expected"],
            "score": score,
            "correct": is_correct
        })

    accuracy = correct / len(test_cases)
    return accuracy, results_summary

# =========================================================
# 📊 SIDEBAR DASHBOARD (NEW UI)
# =========================================================
st.sidebar.title("📊 Evaluation Dashboard")

if st.sidebar.button("Run Evaluation"):
    accuracy, results_summary = run_evaluation()

    st.sidebar.success(f"Accuracy: {accuracy:.2f}")

    st.subheader("🧪 Evaluation Results")

    for r in results_summary:
        if r["correct"]:
            st.markdown(f"✅ **{r['question']}**")
        else:
            st.markdown(f"❌ **{r['question']}**")

        st.markdown(f"- Predicted: {r['prediction']}")
        st.markdown(f"- Expected: {r['expected']}")
        st.markdown(f"- Score: {r['score']:.3f}")
        st.markdown("---")

# =========================================================
# 💬 CHAT INTERFACE
# =========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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
            "❌ I couldn't find a highly confident answer.\n\n"
            "Try rephrasing your question or using different keywords."
        )
    else:
        context = "\n\n".join([
            f"Q: {r['question']}\nA: {r['answer']}"
            for r in results[:3]
        ])

        response = f"""
✅ **Best Answer:**
{best['answer']}

---

🔎 **Related FAQ Context:**
{context}
"""

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.markdown(response)

        with st.expander("🔍 Top Matching FAQs (Debug View)"):
            for r in results:
                st.markdown(f"**Q:** {r['question']}")
                st.markdown(f"**Score:** {r['score']:.3f}")
                st.markdown("---")