import os

import pdfplumber


ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def extract_text_from_pdf(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def extract_text(path):
    extension = os.path.splitext(path)[1].lower()
    if extension == ".pdf":
        return extract_text_from_pdf(path)
    if extension == ".txt":
        return extract_text_from_txt(path)
    raise ValueError("Only PDF and TXT resumes are supported.")
