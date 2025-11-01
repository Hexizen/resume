from django.shortcuts import render
from .utils.feedback import generate_feedback
from .utils.text_extraction import extract_text
from .utils.similarity import get_sbert_model, compute_similarity

# Load SBERT model once (efficient)
model = get_sbert_model()


def index(request):
    """
    Main view — handles resume upload and job description input,
    computes similarity, and generates AI feedback.
    """
    context = {}

    if request.method == "POST":
        resume_file = request.FILES.get("resume")
        job_description = request.POST.get("job_description", "").strip()

        if not resume_file or not job_description:
            context["error"] = "Please upload a resume and enter a job description."
            return render(request, "matcher/index.html", context)

        # Extract text
        try:
            resume_text = extract_text(resume_file)
        except Exception as e:
            context["error"] = f"Failed to extract text: {e}"
            return render(request, "matcher/index.html", context)

        # Compute similarity
        try:
            sim_result = compute_similarity(resume_text, job_description, model)
        except Exception as e:
            context["error"] = f"Similarity computation failed: {e}"
            return render(request, "matcher/index.html", context)

        # Generate feedback using best available engine
        try:
            feedback, model_used = generate_feedback(
                sim_result["matched"],
                sim_result["missing"],
                sim_result["score"]
            )
        except Exception as e:
            feedback, model_used = f"⚠️ Feedback generation failed: {e}", "Error"

        # Display results
        context.update({
            "file_name": resume_file.name,
            "resume_text": resume_text[:2000],
            "job_description": job_description,
            "match_score": sim_result["score"],
            "matched": sim_result["matched"],
            "missing": sim_result["missing"],
            "feedback": feedback,
            "model_used": model_used
        })

        return render(request, "matcher/results.html", context)

    return render(request, "matcher/index.html")
def results(request):
    context = request.session.get('result_context')
    if not context:
        context = {"error": "No results found. Please upload again."}
    return render(request, "matcher/results.html", context)