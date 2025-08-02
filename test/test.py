import os
import csv
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from difflib import SequenceMatcher

from dotenv import load_dotenv
from qdrant_client import QdrantClient

# --- Google Generative AI (paket 'google-generativeai') ---
import google.generativeai as genai
from google.generativeai import configure, embed_content

# ======================================================================
# Konfigurasi
# ======================================================================

load_dotenv()

configure(api_key=os.getenv("GOOGLE_API_KEY"))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "nara_documents")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Model generatif
model = genai.GenerativeModel("gemini-2.0-flash")

# ======================================================================
# Retrieval ke Qdrant
# ======================================================================

def search_docs(query: str, top_k: int = 5) -> List[str]:
    """
    Dapatkan embedding untuk query dan cari dokumen di Qdrant.
    """
    embedding = embed_content(
        content=query,
        task_type="RETRIEVAL_QUERY",
        model="models/embedding-001"
    )["embedding"]

    hits = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=embedding,
        limit=top_k,
    )

    return [hit.payload["text"] for hit in hits if "text" in hit.payload]

# ======================================================================
# Generasi jawaban
# ======================================================================

def ask_rag_ai(query: str, top_k: int = 5) -> str:
    try:
        context = search_docs(query, top_k=top_k)
        joined = "\n\n".join(context) if context else "(Tidak ada konteks yang ditemukan.)"
        prompt = f"""Anda adalah chatbot AI yang ramah dan membantu bernama Nara. Jawab pertanyaan berikut berdasarkan konteks yang diberikan.
Jawab secara akurat dan informatif dalam bahasa Indonesia.

Konteks:
{joined}

Pertanyaan:
{query}
"""
        response = model.generate_content(prompt)
        return getattr(response, "text", "").strip() or "(Tidak ada teks balasan.)"
    except Exception as e:
        return f"Error during RAG AI call: {e}"

def ask_og_ai(query: str) -> str:
    try:
        prompt = f"""Anda adalah chatbot AI yang ramah dan membantu bernama Nara. Jawab pertanyaan berikut.
Jawab secara akurat dan informatif dalam bahasa Indonesia.

Pertanyaan:
{query}
"""
        response = model.generate_content(prompt)
        return getattr(response, "text", "").strip() or "(Tidak ada teks balasan.)"
    except Exception as e:
        return f"Error during Original AI call: {e}"

# ======================================================================
# Evaluator (AI sebagai Judge)
# ======================================================================

def _heuristic_score(expected: str, answer: str) -> float:
    e = (expected or "").strip().lower()
    a = (answer or "").strip().lower()
    if not e or not a:
        return 0.0
    if e in a:
        return 0.95
    return SequenceMatcher(None, e, a).ratio()

def _call_gemini_with_retry(prompt: str, retries: int = 3, backoff: float = 2.0) -> str:
    last_err = None
    for i in range(retries):
        try:
            resp = model.generate_content(prompt)
            return getattr(resp, "text", "")
        except Exception as e:
            last_err = e
            time.sleep(backoff ** i)
    raise RuntimeError(f"Gagal memanggil Gemini setelah {retries} kali: {last_err}")

def judge_with_ai(question: str, expected: str, answer: str) -> Dict[str, str | float]:
    """
    Menilai apakah 'answer' sesuai 'expected' untuk 'question'.
    Return: dict {verdict, score, reason}
    """
    if not expected:
        return {"verdict": "TIDAK PASTI", "score": 0.0, "reason": "Tidak ada jawaban acuan (gold) di CSV."}

    prompt = f"""
Anda adalah evaluator obyektif. Tugas Anda: nilai apakah JAWABAN_KANDIDAT setara secara semantik dengan JAWABAN_ACUAN untuk PERTANYAAN.

KELUARAN HARUS JSON SAJA:
{{
  "verdict": "BENAR|SALAH|TIDAK PASTI",
  "score": 0.0,
  "reason": "alasan singkat <= 30 kata"
}}

Aturan:
- BENAR jika makna utama setara (sinonim/paraferase ok).
- SALAH jika bertentangan/tidak menjawab/fakta inti hilang.
- TIDAK PASTI jika sebagian benar/ambigu.
- Balas HANYA JSON.

PERTANYAAN:
{question}

JAWABAN_ACUAN:
{expected}

JAWABAN_KANDIDAT:
{answer}
"""
    try:
        raw = _call_gemini_with_retry(prompt)
        start = raw.find("{"); end = raw.rfind("}")
        raw_json = raw[start:end+1] if start != -1 and end != -1 else raw
        data = json.loads(raw_json)
        verdict = str(data.get("verdict", "")).upper()
        score = float(data.get("score", 0.0))
        reason = str(data.get("reason", "")).strip()
        if verdict not in ("BENAR", "SALAH", "TIDAK PASTI"):
            score_h = _heuristic_score(expected, answer)
            verdict = "BENAR" if score_h >= 0.8 else ("TIDAK PASTI" if score_h >= 0.6 else "SALAH")
            reason = reason or "Fallback heuristik kemiripan."
            score = score or score_h
        return {"verdict": verdict, "score": round(score, 3), "reason": reason or "(tidak ada alasan)"}
    except Exception:
        score = _heuristic_score(expected, answer)
        verdict = "BENAR" if score >= 0.8 else ("TIDAK PASTI" if score >= 0.6 else "SALAH")
        return {"verdict": verdict, "score": round(score, 3), "reason": "Fallback heuristik (LLM judge gagal)."}

# ======================================================================
# Utilitas CSV
# ======================================================================

def _detect_columns(fieldnames: List[str]) -> Tuple[str, Optional[str]]:
    qcol = None; gcol = None
    if fieldnames:
        lower = {c.lower(): c for c in fieldnames}
        for k in ("pertanyaan", "question", "query"):
            if k in lower: qcol = lower[k]; break
        for k in ("jawaban", "expected", "gold", "label", "answer"):
            if k in lower: gcol = lower[k]; break
    return qcol or fieldnames[0], gcol

def _read_testcases(path: str) -> List[Dict[str, str]]:
    cases: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        peek = f.read(2048); f.seek(0)
        if any(h in peek.lower() for h in ("pertanyaan","question","query","jawaban","expected","gold","label","answer")):
            reader = csv.DictReader(f)
            qcol, gcol = _detect_columns(reader.fieldnames or [])
            for row in reader:
                q = (row.get(qcol) or "").strip()
                g = (row.get(gcol) or "").strip() if gcol else ""
                if q:
                    cases.append({"q": q, "gold": g})
        else:
            reader2 = csv.reader(f)
            for row in reader2:
                if not row: continue
                q = (row[0] or "").strip()
                if q:
                    cases.append({"q": q, "gold": ""})
    return cases

def _open_output_csv(base_dir: str) -> tuple[csv.DictWriter, any, str]:
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(base_dir, f"test-{ts}.csv")
    f = open(output_path, "w", newline="", encoding="utf-8")
    fieldnames = [
        "pertanyaan", "expected",
        "ragAI", "rag_benar", "rag_score", "rag_verdict", "rag_reason",
        "ogAI",  "og_benar",  "og_score",  "og_verdict",  "og_reason",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return writer, f, output_path

# ======================================================================
# Mode batch (dengan jeda per N pertanyaan) & interaktif
# ======================================================================

def _cooldown(seconds: int):
    """Jeda dengan countdown agar terlihat progresnya."""
    if seconds <= 0:
        return
    print(f"Cooldown {seconds} detik untuk menghormati kuota/rate limit...")
    for remaining in range(seconds, 0, -1):
        print(f"\rIstirahat: {remaining:02d}s ", end="", flush=True)
        time.sleep(1)
    print("\rLanjut...         ")

def run_batch_from_csv(
    input_csv_path: str,
    output_dir: str = "test",
    top_k: int = 5,
    threshold: float = 0.7,
    batch_size: int = 5,
    cooldown_seconds: int = 70,
) -> str:
    cases = _read_testcases(input_csv_path)
    if not cases:
        raise ValueError(f"Tidak ada baris terbaca dari '{input_csv_path}'.")

    writer, fh, output_path = _open_output_csv(output_dir)
    try:
        total = len(cases)
        print(f"Menjalankan batch untuk {total} pertanyaan...")
        for i, case in enumerate(cases, start=1):
            q = case["q"]; gold = case.get("gold", "")
            print(f"[{i}/{total}] Memproses: {q[:80]}{'...' if len(q) > 80 else ''}")

            rag_answer = ask_rag_ai(q, top_k=top_k)
            og_answer  = ask_og_ai(q)

            if gold:
                rag_eval = judge_with_ai(q, gold, rag_answer)
                og_eval  = judge_with_ai(q, gold, og_answer)
            else:
                rag_eval = {"verdict": "TIDAK PASTI", "score": 0.0, "reason": "CSV tidak menyediakan expected."}
                og_eval  = {"verdict": "TIDAK PASTI", "score": 0.0, "reason": "CSV tidak menyediakan expected."}

            rag_benar = "TRUE" if (rag_eval["verdict"] == "BENAR" or float(rag_eval["score"]) >= threshold) else "FALSE"
            og_benar  = "TRUE" if (og_eval["verdict"]  == "BENAR" or float(og_eval["score"])  >= threshold) else "FALSE"

            writer.writerow({
                "pertanyaan": q, "expected": gold,
                "ragAI": rag_answer,
                "rag_benar": rag_benar, "rag_score": rag_eval["score"],
                "rag_verdict": rag_eval["verdict"], "rag_reason": rag_eval["reason"],
                "ogAI": og_answer,
                "og_benar": og_benar, "og_score": og_eval["score"],
                "og_verdict": og_eval["verdict"], "og_reason": og_eval["reason"],
            })

            # --- jeda setiap 'batch_size' pertanyaan, kecuali di akhir ---
            if (i % batch_size == 0) and (i < total):
                # pastikan data tersimpan
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except Exception:
                    pass
                _cooldown(cooldown_seconds)

        print(f"Selesai. Hasil disimpan ke {output_path}")
        return output_path
    finally:
        fh.close()

def run_interactive():
    """
    Mode lama: satu pertanyaan, tanpa evaluasi (tetap simpan ke rag_vs_og.csv).
    """
    output_csv_path = 'rag_vs_og.csv'
    question = input("Masukkan pertanyaan Anda: ")
    print("Mendapatkan jawaban dari RAG AI...")
    rag_answer = ask_rag_ai(question)
    print("Mendapatkan jawaban dari Original AI...")
    og_answer = ask_og_ai(question)

    print("" + "="*50)
    print("PERTANYAAN:"); print(question)
    print("="*50)
    print("JAWABAN RAG AI:"); print(rag_answer)
    print("JAWABAN ORIGINAL AI:"); print(og_answer)
    print("" + "="*50)

    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['pertanyaan', 'ragAI', 'ogAI']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'pertanyaan': question, 'ragAI': rag_answer, 'ogAI': og_answer})

    print(f"Hasil telah disimpan di {output_csv_path}")

# ======================================================================
# Entrypoint
# ======================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tester RAG vs OG + AI Judge + Batching Delay")
    parser.add_argument("--batch-csv", help="Path ke file CSV berisi pertanyaan (mis. test.csv)")
    parser.add_argument("--output-dir", default="test", help="Folder output hasil batch (default: test)")
    parser.add_argument("--top-k", type=int, default=5, help="Jumlah dokumen RAG dari Qdrant (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Ambang skor untuk TRUE/FALSE (default: 0.7)")
    parser.add_argument("--batch-size", type=int, default=5, help="Jumlah pertanyaan per batch sebelum jeda (default: 5)")
    parser.add_argument("--cooldown", type=int, default=70, help="Durasi jeda dalam detik setiap selesai satu batch (default: 70)")
    args = parser.parse_args()

    if args.batch_csv:
        out_path = run_batch_from_csv(
            args.batch_csv,
            output_dir=args.output_dir,
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=args.batch_size,
            cooldown_seconds=args.cooldown,
        )
        print(f"Output: {out_path}")
    else:
        run_interactive()
