import streamlit as st
import json
import numpy as np
import warnings
import os
import requests

# Optional heavy deps. sentence-transformers is enough for local query encoding;
# faiss is only needed when building/searching a FAISS index at runtime.
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    HAS_ST = False
    SentenceTransformer = None  # type: ignore[misc, assignment]

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# Suppress FutureWarnings from huggingface_hub
warnings.filterwarnings('ignore', category=FutureWarning)

# -----------------------------
# 1. PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AAU AI Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AAU AI Chat Assistant (FAISS)")
st.caption("Semantic FAQ system powered by Sentence-BERT + FAISS")

# -----------------------------
# 2. LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    if HAS_ST:
        return SentenceTransformer("all-MiniLM-L6-v2")
    return None

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

# If precomputed embeddings and index exist (prebuilt locally), load them.
# This lets Streamlit Cloud avoid heavy installs; create these with build_index.py
EMB_PATH = "embeddings.npy"
NN_PATH = "nn_index.joblib"

def get_secret(key, default=None):
    val = os.environ.get(key)
    if not val:
        try:
            val = st.secrets.get(key, default)
        except FileNotFoundError:
            val = default
    if isinstance(val, str):
        val = val.strip().strip('"').strip("'")
    return val or default


def download_if_missing(url, path, timeout=30):
    """Download `url` to `path` if `path` does not exist. Returns True if file exists.
    Use environment variables or Streamlit secrets to provide URLs.
    """
    if os.path.exists(path):
        return True
    if not url:
        return False
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(1024 * 64):
                if chunk:
                    f.write(chunk)
        return True
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            st.error(
                f"Download failed (404 Not Found): {url}\n\n"
                "Create a GitHub Release and upload the file as a release asset, "
                "or fix the URL in Streamlit secrets."
            )
        else:
            st.warning(f"Failed to download {path}: {e}")
        return False
    except Exception as e:
        st.warning(f"Failed to download {path}: {e}")
        return False

# Try to download precomputed files if URLs are provided (via env or secrets)
emb_url = get_secret("EMB_URL")
nn_url = get_secret("NN_URL")

# Attempt downloads (no-op if files already exist)
emb_downloaded = download_if_missing(emb_url, EMB_PATH)
nn_downloaded = download_if_missing(nn_url, NN_PATH)

# If URLs were provided but the files could not be downloaded, fail fast.
if (emb_url and not emb_downloaded) or (nn_url and not nn_downloaded):
    raise RuntimeError(
        "Failed to download precomputed embeddings/index. "
        "Check your EMB_URL and NN_URL Streamlit secrets and ensure the files are publicly accessible. "
        "If you cannot host the artifacts publicly, use a different object store or upload approach."
    )

if os.path.exists(EMB_PATH) and os.path.exists(NN_PATH):
    embeddings = np.load(EMB_PATH)
    import joblib
    index = joblib.load(NN_PATH)
    index_type = "sklearn"
else:
    embeddings = None
    index = None
    index_type = None

# -----------------------------
# 4. EMBEDDINGS
# -----------------------------
@st.cache_resource
def build_embeddings():
    # 1) If a precomputed embeddings file exists, load it and return immediately.
    if os.path.exists(EMB_PATH):
        return np.load(EMB_PATH)

    # 2) Local model path (fast) when SentenceTransformer is available
    if model is not None:
        return model.encode(questions, normalize_embeddings=True)

    # 3) Fallback: use Hugging Face Inference API for feature-extraction
    hf_token = get_secret('HF_API_TOKEN')
    if not hf_token:
        raise RuntimeError(
            "No HF_API_TOKEN found in environment or Streamlit secrets. "
            "Provide HF_API_TOKEN or host precomputed embeddings and set EMB_URL/NN_URL."
        )

    headers = {"Authorization": f"Bearer {hf_token}"}
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"

    embeddings = []
    try:
        # Batch requests are preferable, but the HF inference pipeline may accept one input at a time.
        for text in questions:
            resp = requests.post(url, headers=headers, json={"inputs": text}, timeout=30)
            resp.raise_for_status()
            emb = resp.json()
            if isinstance(emb, list) and len(emb) and isinstance(emb[0], list):
                emb_vec = np.mean(np.array(emb), axis=0)
            else:
                emb_vec = np.array(emb)
            embeddings.append(emb_vec)
    except requests.exceptions.RequestException as e:
        # Provide a clearer, actionable error to the user rather than a raw ConnectionError trace.
        msg = (
            "Failed to contact the Hugging Face Inference API. "
            "This can be due to missing network/DNS access from Streamlit Cloud or an invalid token. "
            "Recommended actions: (1) Upload precomputed 'embeddings.npy' and 'nn_index.joblib' to a public URL and set EMB_URL/NN_URL in Streamlit Secrets, or (2) add HF_API_TOKEN to Streamlit Secrets and ensure network access. "
            f"(Underlying error: {e})"
        )
        st.error(msg)
        raise RuntimeError(msg)

    return np.array(embeddings)

if embeddings is None:
    embeddings = build_embeddings()

# -----------------------------
# 5. FAISS INDEX
# -----------------------------
@st.cache_resource
def build_index():
    # If faiss available, use it for fast similarity search
    if HAS_FAISS:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(np.array(embeddings))
        return ("faiss", index)

    # Otherwise use scikit-learn NearestNeighbors as a pure-python fallback
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=3, metric='cosine')
    nn.fit(embeddings)
    return ("sklearn", nn)

if index is None:
    index_type, index = build_index()

# -----------------------------
# 6. SEARCH FUNCTION
# -----------------------------
def search(query, top_k=3):
    # Local model: encode query and search with FAISS or sklearn index
    if model is not None:
        query_vec = model.encode([query], normalize_embeddings=True)
        if index_type == "faiss":
            scores, indices = index.search(np.array(query_vec), top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                results.append({
                    "question": questions[idx],
                    "answer": answers[idx],
                    "score": float(score)
                })
            return results

        distances, indices = index.kneighbors(query_vec, n_neighbors=top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                "question": questions[idx],
                "answer": answers[idx],
                "score": 1.0 - float(dist),
            })
        return results

    # Fallback: HF Inference API for query embedding (e.g. Streamlit Cloud without torch)
    hf_token = get_secret('HF_API_TOKEN')
    if not hf_token:
        raise RuntimeError(
            "No HF_API_TOKEN found in environment or Streamlit secrets. "
            "Install sentence-transformers locally (pip install -r requirements.txt) "
            "or set HF_API_TOKEN for cloud deployments without torch."
        )

    headers = {"Authorization": f"Bearer {hf_token}"}
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    try:
        resp = requests.post(url, headers=headers, json={"inputs": query}, timeout=30)
        resp.raise_for_status()
        q_emb = np.array(resp.json())
        if q_emb.ndim == 2:
            q_emb = np.mean(q_emb, axis=0)
    except requests.exceptions.RequestException as e:
        msg = (
            "Failed to get query embedding from Hugging Face Inference API. "
            "Ensure HF_API_TOKEN is set in Streamlit Secrets or host precomputed embeddings and set EMB_URL/NN_URL. "
            f"(Underlying error: {e})"
        )
        st.error(msg)
        raise RuntimeError(msg)

    # sklearn NearestNeighbors returns distances; convert to similarity score
    distances, indices = index.kneighbors([q_emb], n_neighbors=top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        score = 1.0 - float(dist)  # cosine distance -> similarity-like
        results.append({
            "question": questions[idx],
            "answer": answers[idx],
            "score": score
        })
    return results

# -----------------------------
# 7. CHAT MEMORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# 8. DISPLAY CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# 9. USER INPUT (CHAT STYLE)
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
    # FAISS SEARCH
    # -------------------------
    results = search(user_input)
    best = results[0]

    if best["score"] < 0.35:
        response = "❌ Sorry, I couldn't find a relevant answer. Please rephrase your question."
    else:
        response = best["answer"]

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(response)

        # Show top matches (debug / transparency)
        with st.expander("🔍 Top Matching FAQs"):
            for r in results:
                st.markdown(f"**Q:** {r['question']}")
                st.markdown(f"**Score:** {r['score']:.2f}")
                st.markdown("---")
