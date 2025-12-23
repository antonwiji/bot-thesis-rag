import time
from typing import List

import pandas as pd

from models import EvalQuestion, RagEvalResult
from rag_deepseek_chain import build_rag_chain


def precision_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str]):
    if not retrieved_ids:
        return 0.0, 0.0
    relevant_ids = [r for r in relevant_ids if r]
    if not relevant_ids:
        return 0.0, 0.0

    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    inter = len(retrieved_set & relevant_set)

    precision = inter / len(retrieved_ids)
    recall = inter / len(retrieved_set)
    return precision, recall


def load_eval_questions(path: str) -> List[EvalQuestion]:
    df = pd.read_csv(path)
    questions: List[EvalQuestion] = []

    for _, row in df.iterrows():
        relevant_ids: List[str] = []
        if not pd.isna(row["relevant_ids"]):
            relevant_ids = str(row["relevant_ids"]).split("|")

        questions.append(
            EvalQuestion(
                question_id=str(row["question_id"]),
                question=row["question"],
                relevant_ids=relevant_ids,
                category=row.get("category", None),
                note=row.get("note", None),
            )
        )
    return questions


def main():
    qa_chain = build_rag_chain()
    llm_name = "deepseek-chat"

    eval_questions = load_eval_questions("data/eval_questions.csv")

    results: List[RagEvalResult] = []

    for i, q in enumerate(eval_questions, start=1):
        print(f"\n=== [{i}/{len(eval_questions)}] {q.question_id}: {q.question}")

        t0 = time.perf_counter()
        out = qa_chain.invoke({"query": q.question})
        t1 = time.perf_counter()

        latency_sec = t1 - t0
        answer = out["result"]
        source_docs = out["source_documents"]

        retrieved_ids = [
            str(doc.metadata.get("product_id"))
            for doc in source_docs
            if doc.metadata.get("product_id") is not None
        ]

        prec, rec = precision_recall_at_k(retrieved_ids, q.relevant_ids)

        print(f"   Latency={latency_sec:.2f}s | P@k={prec:.2f} | R@k={rec:.2f}")

        res = RagEvalResult(
            question_id=q.question_id,
            question=q.question,
            llm_name=llm_name,
            relevant_ids=q.relevant_ids,
            retrieved_ids=retrieved_ids,
            precision_at_k=prec,
            recall_at_k=rec,
            latency_sec=latency_sec,
            answer=answer,
            top_k=len(retrieved_ids),
        )

        results.append(res)

    # simpan ke CSV
    df_res = pd.DataFrame(
        [
            {
                "question_id": r.question_id,
                "question": r.question,
                "llm_name": r.llm_name,
                "relevant_ids": "|".join(r.relevant_ids),
                "retrieved_ids": "|".join(r.retrieved_ids),
                "precision_at_k": r.precision_at_k,
                "recall_at_k": r.recall_at_k,
                "latency_sec": r.latency_sec,
                "answer": r.answer,
                "top_k": r.top_k,
            }
            for r in results
        ]
    )

    output_path = "data/eval_results_deepseek.csv"
    df_res.to_csv(output_path, index=False, sep=";")
    print(f"\n✅ Hasil evaluasi tersimpan di {output_path}")

    # summary describe
    summary = df_res[["latency_sec", "precision_at_k", "recall_at_k"]].describe()
    print(summary)

    # simpan summary ke file teks
    result_txt_path = "data/result.txt"
    with open(result_txt_path, "w", encoding="utf-8") as f:
        f.write(summary.to_string())

    print(f"\n📝 Ringkasan metrik (describe) tersimpan di {result_txt_path}")


if __name__ == "__main__":
    main()
