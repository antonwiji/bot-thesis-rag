import time
from typing import List, Tuple

import pandas as pd

from models import EvalQuestion, RagEvalResult
from rag_deepseek_chain import build_rag_chain_gpt
# Catatan:
# Pastikan di dalam build_rag_chain() kamu sudah mengatur model LLM ke "gpt-5-nano",
# misalnya ChatOpenAI(model="gpt-5-nano", ...)

# ============================
# KONFIGURASI HARGA TOKEN (USD)
# ============================

# Harga GPT-5 nano per 1 juta token (OpenAI pricing)
# Input:  $0.05 / 1M tokens
# Output: $0.40 / 1M tokens
PRICE_INPUT_PER_M_TOKENS_USD = 0.05   # input tokens GPT-5 nano
PRICE_OUTPUT_PER_M_TOKENS_USD = 0.40  # output tokens GPT-5 nano

# Asumsi konversi karakter -> token (kasar):
# ~1 token ≈ 4 karakter
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Perkiraan jumlah token berdasarkan panjang teks."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_usage(question: str, source_docs, answer: str) -> Tuple[int, int]:
    """
    Estimasi token input dan output untuk satu query:
    - input_tokens ≈ token(question + seluruh context dokumen)
    - output_tokens ≈ token(jawaban)
    """
    # token pertanyaan
    question_tokens = estimate_tokens(question)

    # token context (gabungan isi semua dokumen sumber)
    context_parts = []
    for doc in source_docs:
        content = getattr(doc, "page_content", "")
        if content:
            context_parts.append(content)
    context_text = " ".join(context_parts)
    context_tokens = estimate_tokens(context_text)

    # token jawaban
    answer_tokens = estimate_tokens(answer)

    input_tokens = question_tokens + context_tokens
    output_tokens = answer_tokens
    return input_tokens, output_tokens


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
    # recall dibagi jumlah dokumen relevan
    recall = inter / len(relevant_set)
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
    qa_chain = build_rag_chain_gpt()
    llm_name = "gpt-5-nano"  # ganti nama model di hasil evaluasi

    eval_questions = load_eval_questions("../data/eval_questions.csv")

    results: List[RagEvalResult] = []
    token_stats: List[Tuple[int, int]] = []  # (input_tokens, output_tokens)

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

        # Estimasi token untuk query ini
        input_tokens, output_tokens = estimate_usage(q.question, source_docs, answer)
        token_stats.append((input_tokens, output_tokens))

        print(
            f"   Latency={latency_sec:.2f}s | P@k={prec:.2f} | R@k={rec:.2f} | "
            f"input_tokens≈{input_tokens} | output_tokens≈{output_tokens}"
        )

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
                # kolom tambahan: token & biaya (GPT-5 nano)
                "input_tokens": token_stats[idx][0],
                "output_tokens": token_stats[idx][1],
                "total_tokens": token_stats[idx][0] + token_stats[idx][1],
                "cost_input_usd": (
                    token_stats[idx][0] / 1_000_000 * PRICE_INPUT_PER_M_TOKENS_USD
                ),
                "cost_output_usd": (
                    token_stats[idx][1] / 1_000_000 * PRICE_OUTPUT_PER_M_TOKENS_USD
                ),
            }
            for idx, r in enumerate(results)
        ]
    )

    # total biaya per row
    df_res["cost_total_usd"] = df_res["cost_input_usd"] + df_res["cost_output_usd"]

    # Ganti nama file supaya jelas ini hasil GPT-5 nano
    output_path = "data/eval_results_gpt5_nano.csv"
    df_res.to_csv(output_path, index=False, sep=";")
    print(f"\n✅ Hasil evaluasi tersimpan di {output_path}")

    # summary describe untuk latency & metrik
    summary = df_res[["latency_sec", "precision_at_k", "recall_at_k"]].describe()
    print(summary)

    # summary token & biaya
    total_input_tokens = int(df_res["input_tokens"].sum())
    total_output_tokens = int(df_res["output_tokens"].sum())
    total_cost_usd = df_res["cost_total_usd"].sum()

    print("\n📊 Ringkasan token & biaya (perkiraan) GPT-5 nano:")
    print(f"   Total input tokens  ≈ {total_input_tokens}")
    print(f"   Total output tokens ≈ {total_output_tokens}")
    print(f"   Total biaya         ≈ ${total_cost_usd:.6f} USD")

    # summary biaya + metrik disimpan ke result.txt
    result_txt_path = "data/result_gpt5_nano.txt"
    cost_summary = df_res[
        [
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost_input_usd",
            "cost_output_usd",
            "cost_total_usd",
        ]
    ].agg(["sum", "mean"])

    with open(result_txt_path, "w", encoding="utf-8") as f:
        f.write("Statistik latency & metrik (GPT-5 nano):\n")
        f.write(summary.to_string())
        f.write("\n\nStatistik token & biaya (sum/mean) GPT-5 nano:\n")
        f.write(cost_summary.to_string())
        f.write(
            "\n\nCatatan:\n"
            f"- Harga input per 1M token  (USD): {PRICE_INPUT_PER_M_TOKENS_USD}\n"
            f"- Harga output per 1M token (USD): {PRICE_OUTPUT_PER_M_TOKENS_USD}\n"
            f"- Estimasi token: 1 token ≈ {CHARS_PER_TOKEN} karakter.\n"
        )

    print(f"\n📝 Ringkasan metrik & biaya tersimpan di {result_txt_path}")


if __name__ == "__main__":
    main()
