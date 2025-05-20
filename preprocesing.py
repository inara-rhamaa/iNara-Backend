import os
import glob
import markdown
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from google.generativeai import configure, embed_content
import uuid

# Load environment variables
load_dotenv()

# Konfigurasi API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "nara_documents")

configure(api_key=GOOGLE_API_KEY)

# Inisialisasi Qdrant Cloud
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Cek dan buat koleksi jika belum ada
if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

# Fungsi memotong teks
def chunk_text(text, max_tokens=300):
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk.split()) + len(para.split()) <= max_tokens:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Proses file markdown
def process_markdown_file(path):
    with open(path, "r", encoding="utf-8") as f:
        md_text = f.read()
        html = markdown.markdown(md_text)
        plain_text = html.replace("<p>", "").replace("</p>", "\n")
        return chunk_text(plain_text)

# Embedding via Gemini
def embed_texts(texts):
    return [
        embed_content(
            content=t,
            task_type="RETRIEVAL_DOCUMENT",
            model="models/embedding-001"
        )["embedding"]
        for t in texts
    ]

# Proses semua file
def main():
    files = glob.glob("./data/*.md")
    all_chunks = []
    for path in files:
        chunks = process_markdown_file(path)
        for chunk in chunks:
            all_chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "source": os.path.basename(path)
            })

    print(f"Total chunks: {len(all_chunks)}")

    vectors = embed_texts([c["text"] for c in all_chunks])
    payloads = [{"text": c["text"], "source": c["source"]} for c in all_chunks]
    points = [PointStruct(id=c["id"], vector=v, payload=p) for c, v, p in zip(all_chunks, vectors, payloads)]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print("✅ Sukses: semua data berhasil dimasukkan ke Qdrant Cloud.")

if __name__ == "__main__":
    main()
