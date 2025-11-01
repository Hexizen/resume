import os
import requests
import openai
from dotenv import load_dotenv

MAX_FEEDBACK_LENGTH = 600

load_dotenv()  # Load API keys from .env


def generate_feedback(matched, missing, score):
    """
    Unified feedback generator with fallbacks:
    1. OpenAI GPT (if key available)
    2. HuggingFace text-generation
    3. Offline feedback template
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        openai.api_key = openai_key
        try:
            return generate_feedback_openai(matched, missing, score)
        except Exception:
            pass  # fallback to HuggingFace if fails

    try:
        return generate_feedback_huggingface(matched, missing, score)
    except Exception:
        return generate_feedback_offline(matched, missing, score)


def generate_feedback_openai(matched, missing, score):
    prompt = f"""
    You are an HR resume reviewer.
    Resume match score: {score}%
    Matched keywords: {', '.join(matched[:30])}
    Missing keywords: {', '.join(missing[:30])}
    Write concise, structured feedback (<600 chars):
    Include strengths, weaknesses, and improvement suggestions.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256
    )

    feedback = response["choices"][0]["message"]["content"].strip()
    return feedback[:MAX_FEEDBACK_LENGTH], "OpenAI"


def generate_feedback_huggingface(matched, missing, score):
    HF_API_KEY = os.getenv("HF_API_KEY")
    if not HF_API_KEY:
        raise ValueError("Missing Hugging Face API key")

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": f"Resume score: {score}%. Matched: {matched}. Missing: {missing}. Give feedback under 600 characters."
    }

    response = requests.post(
        "https://api-inference.huggingface.co/models/gpt2",
        headers=headers,
        json=payload
    )
    feedback = response.json()[0]["generated_text"]
    return feedback[:MAX_FEEDBACK_LENGTH], "HuggingFace"


def generate_feedback_offline(matched, missing, score):
    """
    Local fallback if no AI APIs are available.
    """
    feedback = (
        f"Your resume matches {score}% of key job requirements. "
        f"You've covered strengths in areas such as {', '.join(matched[:5])}. "
        f"Consider improving sections like {', '.join(missing[:5])}. "
        f"Tailor your resume to emphasize relevant keywords and measurable results."
    )
    return feedback[:MAX_FEEDBACK_LENGTH], "Offline"