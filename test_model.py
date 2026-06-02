from sentence_transformers import SentenceTransformer

print("Loading...")
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
print("Loaded successfully!")