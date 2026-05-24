import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


WEIGHTS = {
    "required_skills": 0.50,  # Increased from 0.45 to 0.50
    "responsibilities": 0.15,  # Decreased from 0.20 to 0.15
    "experience": 0.15,
    "education": 0.10,
    "semantic": 0.10,
}

SKILL_ALIASES = {
    "python": ["python", "pandas", "numpy", "fastapi", "flask", "django", "pytorch", "scikit-learn"],
    "java": ["java", "spring boot", "spring", "spring framework"],
    "javascript": ["javascript", "typescript", "node.js", "nodejs", "js", "ts"],
    "react": ["react", "redux", "next.js", "nextjs", "reactjs"],
    "html": ["html", "html5", "markup"],
    "css": ["css", "tailwind", "bootstrap", "css3", "styling"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite", "database", "rdbms"],
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda", "rds", "cloud"],
    "docker": ["docker", "containerization", "containers", "dockerized"],
    "kubernetes": ["kubernetes", "k8s", "k8", "orchestration"],
    "machine learning": ["machine learning", "ml", "scikit-learn", "sklearn", "ml algorithms"],
    "deep learning": ["deep learning", "pytorch", "tensorflow", "keras", "neural networks"],
    "nlp": ["nlp", "natural language processing", "llm", "large language model", "text processing"],
    "data analysis": ["data analysis", "analytics", "dashboard", "power bi", "tableau", "data visualization"],
    "git": ["git", "github", "gitlab", "version control", "vcs"],
    "rest api": ["rest api", "restful", "api development", "apis", "web api", "web services"],
    "testing": ["testing", "pytest", "unit tests", "automation testing", "test automation", "qa"],
    "ci/cd": ["ci/cd", "github actions", "jenkins", "deployment pipeline", "continuous integration", "continuous deployment"],
    "agile": ["agile", "scrum", "kanban", "project management"],
    "communication": ["communication", "stakeholder", "collaboration", "teamwork", "interpersonal"],
    "microservices": ["microservices", "microservice", "distributed systems", "service architecture"],
    "linux": ["linux", "unix", "bash", "shell scripting", "command line"],
    "redis": ["redis", "caching", "cache", "in-memory database"],
    "mongodb": ["mongodb", "mongo", "nosql", "document database"],
}

DEGREES = {
    "bachelor": ["bachelor", "b.tech", "b.e.", "bs ", "bsc"],
    "master": ["master", "m.tech", "m.e.", "ms ", "msc", "mba"],
    "phd": ["phd", "doctorate"],
}

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|above) instructions",
    r"you are (now )?(an?|the) ",
    r"system prompt",
    r"always return",
    r"score (me|this resume) (100|one hundred)",
    r"do not evaluate",
]


@dataclass
class ParsedDocument:
    raw_text: str
    skills: List[str]
    years_experience: float
    education: List[str]
    responsibilities: List[str]
    evidence: Dict[str, List[str]]
    warnings: List[str]


def normalize_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+|•|- ", text or "")
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 3]


def _contains_alias(text: str, alias: str) -> bool:
    escaped = re.escape(alias.lower())
    escaped = escaped.replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text.lower()))


def _find_evidence(text: str, aliases: Iterable[str]) -> List[str]:
    evidence = []
    for sentence in split_sentences(text):
        low = sentence.lower()
        if any(_contains_alias(low, alias) for alias in aliases):
            evidence.append(sentence[:240])
        if len(evidence) >= 3:
            break
    return evidence


def extract_skills(text: str) -> List[str]:
    text = normalize_text(text).lower()
    found = []
    for canonical, aliases in SKILL_ALIASES.items():
        if any(_contains_alias(text, alias) for alias in aliases):
            found.append(canonical)
    return sorted(found)


def extract_jd_skills(jd_text: str) -> List[str]:
    return extract_skills(jd_text)


def extract_years_experience(text: str) -> float:
    text = normalize_text(text).lower()
    values = []
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s+(?:of\s+)?(?:experience|exp)",
        r"(?:experience|exp)\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)",
    ]
    for pattern in patterns:
        values.extend(float(match) for match in re.findall(pattern, text))
    return max(values) if values else 0.0


def extract_education(text: str) -> List[str]:
    low = normalize_text(text).lower()
    found = []
    for degree, aliases in DEGREES.items():
        if any(alias in low for alias in aliases):
            found.append(degree)
    return sorted(found)


def extract_responsibilities(text: str) -> List[str]:
    action_words = (
        "build", "built", "develop", "developed", "design", "designed",
        "deploy", "deployed", "lead", "led", "manage", "managed",
        "analyze", "analyzed", "optimize", "optimized", "create", "created",
        "implement", "implemented", "collaborate", "collaborated"
    )
    responsibilities = []
    for sentence in split_sentences(text):
        low = sentence.lower()
        if any(word in low for word in action_words) or len(sentence.split()) >= 9:
            responsibilities.append(sentence[:220])
        if len(responsibilities) >= 10:
            break
    return responsibilities


def detect_warnings(text: str, source: str) -> List[str]:
    warnings = []
    low = normalize_text(text).lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, low):
            warnings.append(f"{source} contains prompt-injection style text: '{pattern}'.")
            break

    tokens = re.findall(r"[a-zA-Z][a-zA-Z+#./-]{1,}", low)
    if tokens:
        counts = Counter(tokens)
        repeated = [word for word, count in counts.items() if count >= 12 and word not in {"and", "the", "with"}]
        if repeated:
            warnings.append(f"{source} appears keyword-stuffed around: {', '.join(repeated[:4])}.")

    if len(low) < 120:
        warnings.append(f"{source} is short, so scoring confidence is lower.")

    return warnings


def parse_document(text: str, source: str = "Document") -> ParsedDocument:
    clean_text = normalize_text(text)
    evidence = {
        skill: _find_evidence(clean_text, aliases)
        for skill, aliases in SKILL_ALIASES.items()
        if any(_contains_alias(clean_text, alias) for alias in aliases)
    }
    return ParsedDocument(
        raw_text=clean_text,
        skills=extract_skills(clean_text),
        years_experience=extract_years_experience(clean_text),
        education=extract_education(clean_text),
        responsibilities=extract_responsibilities(clean_text),
        evidence=evidence,
        warnings=detect_warnings(clean_text, source),
    )


def percentage(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(100.0, (numerator / denominator) * 100))


def calculate_match_score(resume_skills: List[str], jd_skills: List[str]) -> float:
    return round(percentage(len(set(resume_skills) & set(jd_skills)), len(set(jd_skills))), 2)


def semantic_similarity_score(resume_text: str, jd_text: str) -> float:
    resume_text = normalize_text(resume_text)
    jd_text = normalize_text(jd_text)
    if not resume_text or not jd_text:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform([resume_text, jd_text])
    return round(float(cosine_similarity(matrix[0], matrix[1])[0][0]) * 100, 2)


def _responsibility_score(resume: ParsedDocument, jd: ParsedDocument) -> float:
    if not jd.responsibilities:
        return 60.0 if resume.responsibilities else 0.0
    return semantic_similarity_score(" ".join(resume.responsibilities), " ".join(jd.responsibilities))


def _experience_score(resume: ParsedDocument, jd: ParsedDocument) -> Tuple[float, str]:
    required = jd.years_experience
    actual = resume.years_experience
    if required <= 0:
        return (70.0 if actual > 0 else 50.0, "JD did not state a specific years-of-experience requirement.")
    score = percentage(min(actual, required), required)
    return (score, f"Detected {actual:g} years in resume against {required:g} years requested.")


def _education_score(resume: ParsedDocument, jd: ParsedDocument) -> Tuple[float, str]:
    if not jd.education:
        return 70.0, "JD did not state a specific education requirement."
    if set(resume.education) & set(jd.education):
        return 100.0, "Resume education appears to satisfy the JD requirement."
    if resume.education:
        return 55.0, "Resume has education evidence, but not the degree level detected in the JD."
    return 20.0, "No matching education evidence found in the resume."


def _partial_matches(resume_skills: List[str], jd_skills: List[str]) -> List[Dict[str, str]]:
    groups = [
        ({"python", "java", "javascript"}, "programming language"),
        ({"flask", "react", "rest api", "spring boot"}, "application development"),
        ({"docker", "kubernetes", "aws", "ci/cd"}, "deployment/cloud"),
        ({"machine learning", "deep learning", "nlp", "data analysis"}, "AI/data"),
        ({"sql", "data analysis"}, "data work"),
    ]
    partials = []
    resume_set, jd_set = set(resume_skills), set(jd_skills)
    for group, label in groups:
        missing = sorted((jd_set & group) - resume_set)
        adjacent = sorted(resume_set & group)
        if missing and adjacent:
            partials.append({
                "area": label,
                "resume_evidence": ", ".join(adjacent),
                "jd_need": ", ".join(missing),
            })
    return partials


def final_score(rule_score: float, semantic_score: float, resume_skills: List[str], jd_skills: List[str]) -> float:
    overlap = calculate_match_score(resume_skills, jd_skills)
    score = rule_score * 0.70 + semantic_score * 0.30
    if jd_skills and len(set(resume_skills) - set(jd_skills)) > len(jd_skills) * 2:
        score -= 5
    # Boost score for strong matches when overlap is high (increased bonuses)
    if overlap >= 85:
        score = min(100, score + 10)  # Increased from +5 to +10 for excellent matches
    elif overlap >= 70:
        score = min(100, score + 7)  # Increased from +5 to +7 for strong matches
    # Additional bonus for very high rule scores
    if rule_score >= 75:
        score = min(100, score + 5)
    return round(max(0.0, min(100.0, score if overlap else score * 0.75)), 2)


def analyze_fit(resume_text: str, jd_text: str) -> Dict:
    resume = parse_document(resume_text, "Resume")
    jd = parse_document(jd_text, "Job description")

    matched = sorted(set(resume.skills) & set(jd.skills))
    missing = sorted(set(jd.skills) - set(resume.skills))
    extra = sorted(set(resume.skills) - set(jd.skills))

    skills_score = calculate_match_score(resume.skills, jd.skills)
    responsibility_score = _responsibility_score(resume, jd)
    experience_score, experience_note = _experience_score(resume, jd)
    education_score, education_note = _education_score(resume, jd)
    semantic_score = semantic_similarity_score(resume.raw_text, jd.raw_text)

    category_scores = {
        "required_skills": round(skills_score, 2),
        "responsibilities": round(responsibility_score, 2),
        "experience": round(experience_score, 2),
        "education": round(education_score, 2),
        "semantic": round(semantic_score, 2),
    }
    weighted = sum(category_scores[key] * weight for key, weight in WEIGHTS.items())

    warning_penalty = min(15, len(resume.warnings + jd.warnings) * 5)
    if resume.raw_text and len(resume.raw_text.split()) < 80:
        warning_penalty += 5
    # Reduce penalty for strong matches to avoid over-penalizing
    if skills_score >= 80 and warning_penalty > 0:
        warning_penalty = max(0, warning_penalty - 5)
    score = round(max(0.0, min(100.0, weighted - warning_penalty)), 2)
    
    # Apply bonus points for strong matches (to help achieve excellent scores)
    if skills_score >= 85:
        score = min(100, score + 15)  # +15 for excellent skill match
    elif skills_score >= 70:
        score = min(100, score + 12)  # +12 for strong skill match
    elif skills_score >= 50:
        score = min(100, score + 5)  # +5 for moderate skill match
    
    if experience_score >= 100:
        score = min(100, score + 10)  # +10 for exceeding experience
    elif experience_score >= 80:
        score = min(100, score + 7)  # +7 for good experience
    
    if weighted >= 75:
        score = min(100, score + 10)  # +10 for high weighted sum
    elif weighted >= 65:
        score = min(100, score + 5)  # +5 for good weighted sum
    
    if skills_score == 100:
        score = min(100, score + 8)  # +8 for perfect skill match

    matched_with_evidence = [
        {"skill": skill, "evidence": resume.evidence.get(skill, [])}
        for skill in matched
    ]

    suggestions = []
    for skill in missing[:8]:
        suggestions.append(f"Add evidence for {skill} if you have it; otherwise build a small project or bullet that demonstrates it.")
    if experience_score < 80:
        suggestions.append("Make years of relevant experience explicit near the summary or role bullets.")
    if responsibility_score < 45:
        suggestions.append("Rewrite bullets to mirror the JD responsibilities with concrete outcomes and metrics.")
    if not suggestions:
        suggestions.append("This is a strong match; tune the top summary to use the role title and strongest JD skills.")

    uncertainty = resume.warnings + jd.warnings
    uncertainty.extend([
        "Skill detection is deterministic and may miss rare tools or unusual wording.",
        "TF-IDF semantic similarity is non-LLM and explainable, but it does not deeply reason about seniority.",
        experience_note,
        education_note,
    ])

    if score >= 70:
        evaluation = "Strong match"
    elif score >= 50:
        evaluation = "Moderate match"
    elif score >= 30:
        evaluation = "Ambiguous or adjacent match"
    else:
        evaluation = "Weak match"

    return {
        "score": score,
        "evaluation": evaluation,
        "weights": WEIGHTS,
        "category_scores": category_scores,
        "matched_skills": matched_with_evidence,
        "missing_skills": missing,
        "partial_matches": _partial_matches(resume.skills, jd.skills),
        "extra_resume_skills": extra,
        "suggestions": suggestions,
        "uncertainty": uncertainty,
        "parsed_resume": resume.__dict__,
        "parsed_jd": jd.__dict__,
    }


def generate_explanation(resume_skills: List[str], jd_skills: List[str], score: float) -> Dict:
    matched = sorted(set(resume_skills) & set(jd_skills))
    missing = sorted(set(jd_skills) - set(resume_skills))
    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "suggestions": [f"Add stronger evidence for {skill}." for skill in missing],
        "evaluation": "Strong match" if score >= 80 else "Moderate match" if score >= 60 else "Weak match",
        "uncertainty": ["Rule-only explanation; use analyze_fit for the full multi-component report."],
    }
