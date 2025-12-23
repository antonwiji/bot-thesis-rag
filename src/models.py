# src/models.py
from dataclasses import dataclass
from typing import List, Optional


# ==========================
# 1. Data produk (knowledge base) dari CSV
# ==========================

@dataclass
class Product:
    """
    Representasi 1 baris produk dari products.csv
    (title, price, description).
    id digenerate dari nomor baris (1, 2, 3, ...).
    """
    id: str
    title: str
    price: str
    description: str


# ==========================
# 2. Data pertanyaan evaluasi
# ==========================

@dataclass
class EvalQuestion:
    question_id: str
    question: str
    relevant_ids: List[str]

    category: Optional[str] = None
    note: Optional[str] = None


# ==========================
# 3. Hasil evaluasi per pertanyaan
# ==========================

@dataclass
class RagEvalResult:
    question_id: str
    question: str
    llm_name: str

    relevant_ids: List[str]
    retrieved_ids: List[str]

    precision_at_k: float
    recall_at_k: float
    latency_sec: float

    answer: str
    top_k: int
