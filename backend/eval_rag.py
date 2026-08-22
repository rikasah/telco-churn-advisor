"""Evaluate retrieval quality of the RAG store against a small labeled set
of query -> expected source document pairs. Logs hit-rate@3 and MRR to
MLflow (separate experiment from model training) so RAG quality is tracked
the same rigorous way as model quality, not just eyeballed.
"""
import json
import os

import mlflow

import rag

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "rag_eval"

# (query, expected source file) -- expected is matched by filename prefix,
# not exact chunk id, since which chunk within a file is less important
# than retrieving the right document at all.
EVAL_SET = [
    ("Apa saja jenis kontrak yang ditawarkan?", "faq_kontrak.md"),
    ("Bagaimana cara saya membatalkan kontrak bulanan saya?", "faq_kontrak.md"),
    ("Apakah ada denda kalau saya batalkan kontrak 2 tahun lebih awal?", "faq_kontrak.md"),
    ("Metode pembayaran apa saja yang bisa saya pakai?", "faq_kontrak.md"),
    ("Apa saja layanan tambahan yang bisa saya aktifkan?", "faq_layanan.md"),
    ("Apa bedanya Fiber optic sama DSL?", "faq_layanan.md"),
    ("Apakah Tech Support bisa membantu mencegah saya berhenti berlangganan?", "faq_layanan.md"),
    ("Siapa yang termasuk pelanggan berisiko tinggi churn?", "kebijakan_retensi.md"),
    ("Diskon apa yang ditawarkan ke pelanggan yang berisiko pindah?", "kebijakan_retensi.md"),
    ("Apakah tim bisa otomatis memberikan diskon retensi tanpa persetujuan pelanggan?", "kebijakan_retensi.md"),
]


def evaluate(k: int = 3) -> dict:
    per_query = []
    hits = 0
    reciprocal_ranks = []

    for query, expected_doc in EVAL_SET:
        results = rag.retrieve(query, k=k)
        retrieved_docs = [r["source"].split("#")[0] for r in results]

        rank = None
        for i, doc in enumerate(retrieved_docs, start=1):
            if doc == expected_doc:
                rank = i
                break

        hit = rank is not None
        hits += int(hit)
        reciprocal_ranks.append(1.0 / rank if hit else 0.0)

        per_query.append(
            {
                "query": query,
                "expected": expected_doc,
                "retrieved": retrieved_docs,
                "hit": hit,
                "rank": rank,
            }
        )

    n = len(EVAL_SET)
    metrics = {
        f"hit_rate_at_{k}": hits / n,
        "mrr": sum(reciprocal_ranks) / n,
        "n_queries": n,
    }
    return {"metrics": metrics, "per_query": per_query}


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    result = evaluate()
    metrics = result["metrics"]

    with mlflow.start_run(run_name="rag_eval"):
        mlflow.log_metric("hit_rate_at_3", metrics["hit_rate_at_3"])
        mlflow.log_metric("mrr", metrics["mrr"])
        mlflow.log_param("n_queries", metrics["n_queries"])

        report_path = "/tmp/rag_eval_report.json"
        with open(report_path, "w") as f:
            json.dump(result["per_query"], f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(report_path, artifact_path="rag_eval")

    print(f"hit_rate@3={metrics['hit_rate_at_3']:.2f}  mrr={metrics['mrr']:.2f}")
    for q in result["per_query"]:
        status = "OK " if q["hit"] else "MISS"
        print(f"[{status}] rank={q['rank']} expected={q['expected']:<22} query={q['query']}")


if __name__ == "__main__":
    main()
