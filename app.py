import streamlit as st
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AAU Hybrid AI Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AAU Hybrid FAQ Chatbot (BM25 + FAISS + Intent Classifier)")
st.caption("Semantic + Keyword + Context-aware Retrieval System")

# =========================================================
# 2. LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

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
# 4. BM25 PREPROCESSING
# =========================================================
def tokenize(text):
    return text.lower().split()

tokenized_corpus = [tokenize(q) for q in questions]
bm25 = BM25Okapi(tokenized_corpus)

# =========================================================
# 5. QUERY EXPANSION
# =========================================================
def expand_query(query):
    synonyms = {
        "admission": "admission entry requirement apply university",
        "apply": "application admission requirement",
        "fee": "tuition cost payment price",
        "deadline": "due date submission deadline",
        "exam": "entrance test assessment"
    }

    for k, v in synonyms.items():
        if k in query.lower():
            query += " " + v

    return query

# =========================================================
# 6. CHAT MEMORY
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# 7. INTENT CLASSIFIER
# =========================================================
train_data = [
    ("What is the fee?", "standalone"),
    ("How do I apply?", "standalone"),
    ("Where is the office?", "standalone"),
    ("What about it?", "follow_up"),
    ("And then?", "follow_up"),
    ("How about that?", "follow_up"),
]

texts = [t[0] for t in train_data]
labels = [t[1] for t in train_data]

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

@st.cache_resource
def train_clf():
    X = model.encode(texts, normalize_embeddings=True)
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf

clf = train_clf()

def predict_intent(q):
    vec = model.encode([q], normalize_embeddings=True)
    pred = clf.predict(vec)[0]
    return encoder.inverse_transform([pred])[0]

def is_followup(q):
    return predict_intent(q) == "follow_up"

# =========================================================
# 8. CONTEXT HANDLING
# =========================================================
def build_context(q):
    if not is_followup(q):
        return q

    if len(st.session_state.history) == 0:
        return q

    return " ".join(st.session_state.history[-2:]) + " " + q

# =========================================================
# 9. EMBEDDINGS + FAISS
# =========================================================
@st.cache_resource
def build_embeddings():
    return model.encode(questions, normalize_embeddings=True)

embeddings = build_embeddings()

@st.cache_resource
def build_faiss():
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))
    return index

index = build_faiss()

# =========================================================
# 10. BM25 SEARCH
# =========================================================
def bm25_search(query, top_k=10):
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)

    idxs = np.argsort(scores)[::-1][:top_k]

    return {
        i: scores[i] for i in idxs
    }

# =========================================================
# 11. FAISS SEARCH
# =========================================================
def faiss_search(query, top_k=10):
    vec = model.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(np.array(vec), top_k)

    return {
        i: float(s) for i, s in zip(idxs[0], scores[0])
    }

# =========================================================
# 12. HYBRID SEARCH (BM25 + FAISS)
# =========================================================
def hybrid_search(query, top_k=5, alpha=0.7):

    query = query.lower().strip()
    query = build_context(query)
    query = expand_query(query)

    bm25_res = bm25_search(query)
    faiss_res = faiss_search(query)

    all_idx = set(bm25_res.keys()) | set(faiss_res.keys())

    results = []

    for i in all_idx:
        bm = bm25_res.get(i, 0)
        fs = faiss_res.get(i, 0)

        score = alpha * fs + (1 - alpha) * bm

        results.append({
            "question": questions[i],
            "answer": answers[i],
            "score": score
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

# =========================================================
# 13. CONFIDENCE
# =========================================================
def confident(res):
    return len(res) > 0 and res[0]["score"] > 0.35

# =========================================================
# 14. EVALUATION DATASET
# =========================================================
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
        "expected": "Tuition fees are usually paid in installments per semester, typically two installments per semester or four per year depending on the program.",
        "evaluation_level": 2,
        "evaluation_name": "Paraphrased Questions",
        "purpose": "Tests semantic understanding of reworded questions."
    }
]

# =========================================================
# 15. EVALUATION FUNCTIONS
# =========================================================
def semantic_score(a, b):
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def run_evaluation():
    correct = 0
    results_summary = []

    for t in test_cases:
        pred_results = hybrid_search(t["question"])
        prediction = pred_results[0]["answer"] if len(pred_results) > 0 else "No answer found"
        
        score = semantic_score(prediction, t["expected"])
        is_correct = score >= 0.6  #
        if is_correct:
            correct += 1

        results_summary.append({
            "question": t["question"],
            "prediction": prediction,
            "expected": t["expected"],
            "score": score,
            "correct": is_correct,
            "evaluation_level": t.get("evaluation_level", "N/A"),
            "evaluation_name": t.get("evaluation_name", "N/A"),
            "purpose": t.get("purpose", "N/A")
        })

    accuracy = correct / len(test_cases)
    return accuracy, results_summary

# =========================================================
# 16. SIDEBAR EVALUATION DASHBOARD
# =========================================================
st.sidebar.title("📊 Evaluation Dashboard")
st.sidebar.write("Evaluate the hybrid retrieval model against various search complexities.")

if st.sidebar.button("Run System Evaluation"):
    accuracy, results_summary = run_evaluation()
    
    st.sidebar.success(f"Accuracy: {accuracy * 100:.1f}%")
    
    st.markdown("## 🧪 System Evaluation Results")
    st.markdown(f"**Overall Accuracy:** `{accuracy * 100:.1f}%` ({sum(1 for r in results_summary if r['correct'])} / {len(results_summary)} test cases passed)")
    
    for r in results_summary:
        status_icon = "✅" if r["correct"] else "❌"
        
        with st.container():
            st.markdown(f"### {status_icon} Level {r['evaluation_level']}: {r['evaluation_name']}")
            st.markdown(f"**Purpose:** *{r['purpose']}*")
            st.markdown(f"**Test Question:** `{r['question']}`")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Expected Answer:**\n{r['expected']}")
            with col2:
                if r["correct"]:
                    st.success(f"**Predicted Answer:**\n{r['prediction']}")
                else:
                    st.error(f"**Predicted Answer:**\n{r['prediction']}")
            
            st.markdown(f"**Semantic Similarity Score:** `{r['score']:.4f}` (Threshold: `>= 0.6`)")
            st.markdown("---")


# =========================================================
# 14. CHAT HISTORY DISPLAY
# =========================================================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =========================================================
# 15. USER INPUT
# =========================================================
user_input = st.chat_input("Ask AAU questions...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.history.append(user_input)

    results = hybrid_search(user_input)
    best = results[0]

    if not confident(results):
        response = "❌ No confident answer found. Try rephrasing."
    else:
        context = "\n\n".join(
            f"Q: {r['question']}\nA: {r['answer']}"
            for r in results[:3]
        )

        response = f"""
✅ **Answer:**
{best['answer']}

---

🔎 **Hybrid Context (BM25 + FAISS):**
{context}
"""

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)

        with st.expander("🔍 Debug Scores"):
            for r in results:
                st.markdown(f"**Q:** {r['question']}")
                st.markdown(f"Score: {r['score']:.3f}")
                st.markdown("---")

