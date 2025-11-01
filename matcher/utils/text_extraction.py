import os
import fitz  # PyMuPDF
import docx2txt
from PIL import Image
import pytesseract


def extract_text(file):
    """Extract text from PDF, DOCX, TXT, or Image resume."""
    ext = os.path.splitext(file.name)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file)
    elif ext == ".docx":
        return extract_text_from_docx(file)
    elif ext == ".txt":
        return extract_text_from_txt(file)
    elif ext in [".jpg", ".jpeg", ".png"]:
        return extract_text_from_image(file)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text("text")
    return text.strip()


def extract_text_from_docx(file):
    return docx2txt.process(file)


def extract_text_from_txt(file):
    return file.read().decode("utf-8", errors="ignore")


def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image).strip()