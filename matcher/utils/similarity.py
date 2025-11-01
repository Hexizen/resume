from sentence_transformers import SentenceTransformer, util
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_cache = {}


def get_sbert_model(model_name="all-MiniLM-L6-v2"):
    if model_name not in model_cache:
        model_cache[model_name] = SentenceTransformer(model_name, device=DEVICE)
    return model_cache[model_name]


def compute_similarity(resume_text, jd_text, model):
    """Compute keyword-based + semantic similarity score."""
    jd_keywords = [w.lower().strip(",.()") for w in jd_text.split()]
    matched = [kw for kw in jd_keywords if kw in resume_text.lower()]
    missing = [kw for kw in jd_keywords if kw not in resume_text.lower()]

    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    cosine_score = util.cos_sim(resume_emb, jd_emb).item()

    keyword_score = len(matched) / len(jd_keywords) if jd_keywords else 0
    combined_score = round(((keyword_score * 0.6) + (cosine_score * 0.4)) * 100, 2)

    return {"score": combined_score, "matched": matched, "missing": missing}