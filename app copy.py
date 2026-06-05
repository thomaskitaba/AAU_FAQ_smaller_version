import streamlit as st
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AAU AI Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AAU AI Chat Assistant (FAISS + Intent Classifier)")
st.caption("Semantic FAQ + Follow-up Detection + Evaluation System")

# =========================================================
# 2. LOAD MODEL (SBERT)
# =========================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =========================================================
# 3. LOAD FAQ DATA
# =========================================================
@st.cache_data
def load_data():
    with open("faq_data.json", "r", encoding="utf-8") as f:
        return json.load(f)

faq_data = load_data()

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# =========================================================
# 4. QUERY EXPANSION
# =========================================================
def expand_query(query):
    synonyms = {
        "admission": "admission entry requirement apply university",
        "apply": "application admission entry requirement",
        "exam": "entrance exam requirement test",
        "fee": "payment tuition cost price",
        "deadline": "due date submission deadline"
    }

    for k, v in synonyms.items():
        if k in query.lower():
            query += " " + v

    return query

# =========================================================
# 5. CHAT MEMORY
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# 6. INTENT CLASSIFIER (TRAINING DATA)
# =========================================================
train_data = [
    ("What is the application fee for graduate programs?", "standalone"),
    ("How do I apply for AAU graduate programs?", "standalone"),
    ("What is the tuition fee?", "standalone"),
    ("When is the payment deadline?", "standalone"),
    ("Where is the admission office?", "standalone"),

    ("How much does it cost?", "follow_up"),
    ("What about the fee?", "follow_up"),
    ("Where should I submit them?", "follow_up"),
    ("And after that?", "follow_up"),
    ("How do I pay it?", "follow_up"),
    ("What happens next?", "follow_up"),
]

texts = [t[0] for t in train_data]
labels = [t[1] for t in train_data]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

@st.cache_resource
def train_classifier():
    X = model.encode(texts, normalize_embeddings=True)
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf

classifier = train_classifier()

def predict_intent(question):
    vec = model.encode([question], normalize_embeddings=True)
    pred = classifier.predict(vec)[0]
    return label_encoder.inverse_transform([pred])[0]

def is_follow_up(question):
    return predict_intent(question) == "follow_up"

# =========================================================
# 7. CONTEXT BUILDER
# =========================================================
def build_context_query(question):
    if not is_follow_up(question):
        return question

    if len(st.session_state.history) == 0:
        return question

    context = " ".join(st.session_state.history[-2:])
    return f"{context} {question}"

# =========================================================
# 8. EMBEDDINGS + FAISS
# =========================================================
@st.cache_resource
def build_embeddings():
    return model.encode(questions, normalize_embeddings=True)

embeddings = build_embeddings()

@st.cache_resource
def build_index():
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))
    return index

index = build_index()

# =========================================================
# 9. SEARCH FUNCTION
# =========================================================
def search(query, top_k=5):
    query = query.lower().strip()

    # intent-aware context injection
    query = build_context_query(query)

    # expansion
    query = expand_query(query)

    vec = model.encode([query], normalize_embeddings=True)

    scores, idxs = index.search(np.array(vec), top_k)

    results = []
    for score, i in zip(scores[0], idxs[0]):
        results.append({
            "question": questions[i],
            "answer": answers[i],
            "score": float(score)
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)

# =========================================================
# 10. CONFIDENCE
# =========================================================
def is_confident(results):
    return len(results) > 0 and results[0]["score"] > 0.35

# =========================================================
# 11. EVALUATION DATASET
# =========================================================
# test_cases = [
#     {
#         "question": "How much do I need to pay for graduate application?",
#         "expected": "The application fee for Graduate Speciality programs is ETB 500.",
#         "evaluation_level": 1,
#         "evaluation_name": "Exact FAQ Match",
#         "purpose": "Direct retrieval test"
#     },
#     {
#         "question": "Where should I submit my documents?",
#         "expected": "AAU Admission Office, Main Campus Registrar Building, Room 203.",
#         "evaluation_level": 4,
#         "evaluation_name": "Multi-Hop",
#         "purpose": "Indirect reasoning test"
#     },
#     {
#         "question": "Can I reuse my admission test result?",
#         "expected": "No. The Undergraduate Admission Test (UAT) result is valid only for the specific cycle.",
#         "evaluation_level": 3,
#         "evaluation_name": "Keyword Poor",
#         "purpose": "Robust retrieval test"
#     }
# ]
test_cases = [
    {
        "question": "How much do I need to pay for graduate application?",
        "expected": "The application fee for Graduate Speciality programs is ETB 500.",
        "evaluation_level": 1,
        "evaluation_name": "Exact FAQ Match",
        "purpose": "Tests retrieval when the user asks a question that closely matches an existing FAQ."
    },
    {
        "question": "Where should I submit my documents?",
        "expected": "AAU Admission Office, Main Campus Registrar Building, Room 203.",
        "evaluation_level": 4,
        "evaluation_name": "Multi-Hop / Related Information",
        "purpose": "Tests indirect reasoning about document submission location."
    },
    {
        "question": "Can I reuse my admission test result?",
        "expected": "No. The Undergraduate Admission Test (UAT) result is valid only for the specific application cycle.",
        "evaluation_level": 3,
        "evaluation_name": "Keyword-Poor Questions",
        "purpose": "Tests robustness when important keywords are missing or replaced."
    },
    {
        "question": "What is the tuition fee payment method?",
        "expected": "Tuition fees are usually paid in installments per semester.",
        "evaluation_level": 2,
        "evaluation_name": "Paraphrased Questions",
        "purpose": "Tests semantic understanding of reworded questions."
    }
]
# =========================================================
# 12. EVALUATION
# =========================================================
def semantic_score(a, b):
    e1 = model.encode(a, convert_to_tensor=True)
    e2 = model.encode(b, convert_to_tensor=True)
    return util.cos_sim(e1, e2).item()

def run_evaluation():
    correct = 0
    results = []

    for t in test_cases:
        pred = search(t["question"])[0]["answer"]
        score = semantic_score(pred, t["expected"])

        ok = score >= 0.75
        if ok:
            correct += 1

        results.append({**t, "prediction": pred, "score": score, "correct": ok})

    return correct / len(test_cases), results

# =========================================================
# 13. SIDEBAR DASHBOARD
# =========================================================
st.sidebar.title("📊 Evaluation")

if st.sidebar.button("Run Evaluation"):
    acc, res = run_evaluation()
    st.sidebar.success(f"Accuracy: {acc:.2f}")

    for r in res:
        st.markdown("✅" if r["correct"] else "❌")
        st.markdown(f"Q: {r['question']}")
        st.markdown(f"Pred: {r['prediction']}")
        st.markdown(f"Expected: {r['expected']}")
        st.markdown(f"Score: {r['score']:.3f}")
        st.markdown(f"Level: {r['evaluation_level']} - {r['evaluation_name']}")
        st.markdown(f"Purpose: {r['purpose']}")
        st.markdown("---")

# =========================================================
# 14. CHAT UI
# =========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about AAU...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.history.append(user_input)

    results = search(user_input)
    best = results[0]

    if not is_confident(results):
        response = "❌ No confident answer found. Try rephrasing."
    else:
        context = "\n\n".join([
            f"Q: {r['question']}\nA: {r['answer']}"
            for r in results[:3]
        ])

        response = f"""
✅ **Answer:**
{best['answer']}

---

🔎 **Context:**
{context}
"""

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)

        # Debug section
        with st.expander("🔍 Top Matching FAQs (Debug View)"):
            for r in results:
                st.markdown(f"**Q:** {r['question']}")
                st.markdown(f"**Score:** {r['score']:.3f}")
                st.markdown("---")