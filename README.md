# Resume Fit Scorer

A Flask web application that scores how well a resume fits a job description. It is built as a multi-component pipeline rather than a single LLM response:

1. Text extraction from PDF or TXT resume files.
2. Structured parsing for skills, evidence snippets, responsibilities, experience, education, and warnings.
3. Deterministic scoring for required skill overlap, evidence, years of experience, and education.
4. Non-LLM semantic similarity using TF-IDF cosine similarity.
5. Explainable reporting with matched skills, gaps, partial matches, suggestions, and uncertainty.

## Why the components are split

The deterministic parser and rules sit at inference time because the score needs to be auditable. A job seeker can see which skills were matched and why points were lost. TF-IDF provides a lightweight non-LLM semantic component that catches overlap beyond exact keywords without depending on model downloads or opaque generation.

## Score weights

- Required skills: 45%
- Responsibility similarity: 20%
- Years of experience: 15%
- Education: 10%
- Overall semantic similarity: 10%

Warnings for short, noisy, keyword-stuffed, or prompt-injected text apply a small penalty and are shown in the uncertainty section.

## Run

```bash
env\Scripts\activate
python app.py
```

Open `http://127.0.0.1:5000`.

## Test cases

Run:

```bash
env\Scripts\python run_tests.py
```

Included cases:

- Strong match: backend resume against backend JD.
- Poor match: retail resume against backend JD.
- Ambiguous case: data analyst switching toward ML engineering.
- Adversarial case: keyword-stuffed resume with prompt-injection text.
