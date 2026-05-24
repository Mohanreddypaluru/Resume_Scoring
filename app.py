import os
from uuid import uuid4

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from parser.resume_parser import ALLOWED_EXTENSIONS, extract_text
from scorer.scorer import analyze_fit


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    resume = request.files.get("resume")
    jd = request.form.get("jd", "").strip()

    if not resume or not resume.filename:
        return render_template("index.html", error="Please upload a PDF or TXT resume.", jd=jd)
    if not allowed_file(resume.filename):
        return render_template("index.html", error="Only PDF and TXT resumes are supported.", jd=jd)
    if not jd:
        return render_template("index.html", error="Please paste a job description.", jd=jd)

    filename = f"{uuid4().hex}_{secure_filename(resume.filename)}"
    resume_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    resume.save(resume_path)

    try:
        resume_text = extract_text(resume_path)
        result = analyze_fit(resume_text, jd)
    except Exception as exc:
        return render_template("index.html", error=f"Could not analyze the resume: {exc}", jd=jd)

    return render_template("index.html", result=result, jd=jd)


if __name__ == "__main__":
    app.run(debug=True)
