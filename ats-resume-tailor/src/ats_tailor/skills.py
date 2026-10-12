"""Словарь навыков, синонимы и поиск ключевых слов в тексте.

Каждый канонический навык имеет список вариантов написания (EN/RU). Поиск идёт по
нижнему регистру с «границами слова», которые корректно работают для C++, C#, CI/CD.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Канонический навык -> варианты написания (канонический добавляется автоматически).
SYNONYMS: dict[str, list[str]] = {
    "Python": ["python3", "python 3", "питон"],
    "LLM": [
        "llms",
        "large language model",
        "large language models",
        "большие языковые модели",
        "большими языковыми моделями",
        "языковые модели",
        "языковыми моделями",
    ],
    "RAG": ["retrieval-augmented generation", "retrieval augmented generation"],
    "NLP": ["natural language processing", "обработка естественного языка"],
    "Machine Learning": ["ml", "машинное обучение", "машинного обучения"],
    "Deep Learning": ["глубокое обучение", "глубокого обучения"],
    "Computer Vision": ["cv", "компьютерное зрение"],
    "Prompt Engineering": ["prompt-engineering", "промпт-инжиниринг", "промпт инжиниринг"],
    "Fine-tuning": ["fine tuning", "finetuning", "lora", "qlora", "дообучение", "файнтюнинг"],
    "AI Agents": [
        "ai agent",
        "llm agents",
        "llm-агенты",
        "llm-агентов",
        "ai-агенты",
        "ai-агентов",
        "ии-агенты",
        "ии-агентов",
        "агентные системы",
        "агентных систем",
    ],
    "Embeddings": ["эмбеддинги", "embedding", "векторные представления"],
    "Vector DB": [
        "vector database",
        "vector databases",
        "vector db",
        "векторные базы данных",
        "векторные бд",
        "векторная бд",
        "pgvector",
        "qdrant",
        "milvus",
        "weaviate",
        "chroma",
        "faiss",
        "pinecone",
    ],
    "PyTorch": ["torch"],
    "TensorFlow": ["tf", "keras"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "Hugging Face": ["huggingface", "hugging face transformers", "hf transformers"],
    "LangChain": ["langgraph"],
    "LlamaIndex": ["llama-index", "llama index"],
    "OpenAI API": ["openai", "chatgpt api", "gpt-4", "gpt-4o"],
    "vLLM": [],
    "Ollama": [],
    "FastAPI": ["fast api"],
    "Django": [],
    "Flask": [],
    "REST API": ["rest", "restful", "rest api", "rest-api"],
    "gRPC": [],
    "SQL": [],
    "PostgreSQL": ["postgres", "postgresql", "psql", "постгрес"],
    "MySQL": [],
    "MongoDB": ["mongo"],
    "Redis": [],
    "Elasticsearch": ["elastic", "opensearch"],
    "Kafka": ["apache kafka"],
    "RabbitMQ": ["rabbit mq"],
    "Celery": [],
    "Docker": ["docker compose", "docker-compose", "докер"],
    "Kubernetes": ["k8s", "кубернетес"],
    "CI/CD": ["ci cd", "ci-cd", "github actions", "gitlab ci", "jenkins"],
    "Git": ["github", "gitlab"],
    "Linux": ["unix", "bash"],
    "AWS": ["amazon web services", "sagemaker"],
    "GCP": ["google cloud", "google cloud platform", "vertex ai"],
    "Azure": ["microsoft azure"],
    "MLflow": ["ml flow"],
    "MLOps": ["ml ops"],
    "Airflow": ["apache airflow"],
    "Spark": ["pyspark", "apache spark"],
    "Pandas": [],
    "NumPy": ["numpy"],
    "Grafana": [],
    "Prometheus": [],
    "A/B testing": [
        "a/b",
        "a/b-тесты",
        "a/b-тестов",
        "a/b тесты",
        "a/b тестов",
        "a/b-тестирование",
        "a/b тестирование",
        "ab testing",
        "ab-тесты",
    ],
    "Statistics": ["статистика", "математическая статистика"],
    "English": ["английский", "английского", "английским", "english b2", "english c1"],
    "Agile": ["scrum", "kanban"],
    "TypeScript": ["ts"],
    "JavaScript": ["js"],
    "Go": ["golang"],
    "Java": [],
    "C++": ["cpp"],
    "C#": [],
}

# Очень короткие/двусмысленные варианты, которые ищем только с учётом регистра в исходнике.
_CASE_SENSITIVE_VARIANTS: set[str] = {"ml", "cv", "tf", "ts", "js", "go", "rest"}

_BOUNDARY_LEFT = r"(?<![\w+#])"
_BOUNDARY_RIGHT = r"(?![\w+#])"


def _pattern_for(variant: str, case_sensitive: bool) -> re.Pattern[str]:
    escaped = re.escape(variant).replace(r"\ ", r"[\s\-]+")
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(_BOUNDARY_LEFT + escaped + _BOUNDARY_RIGHT, flags)


@lru_cache(maxsize=1)
def _compiled() -> dict[str, list[re.Pattern[str]]]:
    compiled: dict[str, list[re.Pattern[str]]] = {}
    for canon, variants in SYNONYMS.items():
        pats: list[re.Pattern[str]] = []
        for v in [canon, *variants]:
            v_low = v.lower()
            if v_low in _CASE_SENSITIVE_VARIANTS:
                # «ML», «CV», «Go» — только в исходном регистре, иначе ложные срабатывания.
                pats.append(_pattern_for(v.upper() if v_low != "go" else "Go", True))
            else:
                pats.append(_pattern_for(v, False))
        compiled[canon] = pats
    return compiled


_CANON_BY_LOWER: dict[str, str] = {}
for _canon, _variants in SYNONYMS.items():
    _CANON_BY_LOWER[_canon.lower()] = _canon
    for _v in _variants:
        _CANON_BY_LOWER.setdefault(_v.lower(), _canon)


def canonicalize(term: str) -> str:
    """Привести произвольное написание навыка к каноническому (если известно)."""
    t = term.strip()
    return _CANON_BY_LOWER.get(t.lower(), t)


def variants(term: str) -> list[str]:
    canon = canonicalize(term)
    return [canon, *SYNONYMS.get(canon, [])]


def find_skills(text: str) -> set[str]:
    """Все известные канонические навыки, упомянутые в тексте."""
    found: set[str] = set()
    for canon, pats in _compiled().items():
        if any(p.search(text) for p in pats):
            found.add(canon)
    return found


def contains_keyword(text: str, keyword: str) -> bool:
    """Есть ли ключевое слово (с учётом синонимов) в тексте."""
    canon = canonicalize(keyword)
    pats = _compiled().get(canon)
    if pats is None:
        pats = [_pattern_for(keyword, False)]
    return any(p.search(text) for p in pats)


# Технические токены вне словаря: CamelCase, АББРЕВИАТУРЫ, с цифрами/+/#.
_TECH_TOKEN = re.compile(r"(?<![\w.])([A-Za-z][A-Za-z0-9]*(?:[+#./-][A-Za-z0-9+#]+)*\+*)(?![\w])")
_TECH_STOP = {
    "I",
    "A",
    "AI",
    "IT",
    "OK",
    "HR",
    "CV",
    "ML",
    "US",
    "EU",
    "UK",
    "RU",
    "PM",
    "QA",
    "TBD",
    "API",
    "SDK",
    "UI",
    "UX",
    "B2B",
    "B2C",
    "CEO",
    "CTO",
    "KPI",
    "OKR",
    "PhD",
    "MSc",
    "BSc",
}


def find_tech_tokens(text: str) -> set[str]:
    """Технологии, которых нет в словаре: «ClickHouse», «MLflow», «ONNX», «TensorRT»."""
    out: set[str] = set()
    for m in _TECH_TOKEN.finditer(text):
        tok = m.group(1).rstrip(".-/")
        if len(tok) < 2 or tok in _TECH_STOP:
            continue
        has_inner_upper = any(c.isupper() for c in tok[1:])
        is_acronym = tok.isupper() and len(tok) >= 2
        has_special = any(c.isdigit() or c in "+#" for c in tok)
        if not (has_inner_upper or is_acronym or has_special):
            continue
        canon = canonicalize(tok)
        if canon in SYNONYMS:
            continue  # уже учтено словарём
        if tok.isdigit():
            continue
        out.add(tok)
    return out


def normalize_for_matching(text: str) -> str:
    """Заменить синонимы на канонический токен (для TF-IDF), например python3 -> python."""
    out = text
    for canon, pats in _compiled().items():
        token = " " + re.sub(r"[^\w+#]+", "_", canon.lower()) + " "
        for p in pats:
            out = p.sub(token, out)
    return out
