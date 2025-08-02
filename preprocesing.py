import os
import glob
import markdown
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from google.generativeai import configure, embed_content
import uuid
import time

# Load API Key
load_dotenv()
configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Qdrant Setup
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "nara_documents")

# Cek & buat koleksi
if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

def chunk_text(text, max_tokens=300):
    paragraphs = text.split("\n\n")
    chunks, current_chunk = [], ""
    for para in paragraphs:
        if len(current_chunk.split()) + len(para.split()) <= max_tokens:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def process_markdown_file(path):
    with open(path, "r", encoding="utf-8") as f:
        md_text = f.read()
        html = markdown.markdown(md_text)
        plain_text = html.replace("<p>", "").replace("</p>", "\n")
        return chunk_text(plain_text)

def embed_texts(texts, batch_size=20):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_vectors = [
                embed_content(
                    content=t,
                    task_type="RETRIEVAL_DOCUMENT",
                    model="models/embedding-001"
                )["embedding"]
                for t in batch
            ]
            vectors.extend(batch_vectors)
        except Exception as e:
            print(f"⚠️ Gagal proses batch {i} - {i + batch_size}: {e}")
            # Opsi retry ringan
            time.sleep(1)
    return vectors


def main():
    files = glob.glob("./data/**/*.md", recursive=True)
    all_chunks = []
    for path in files:
        chunks = process_markdown_file(path)
        for chunk in chunks:
            all_chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "source": os.path.relpath(path, "./data")
            })

    print(f"Total chunks: {len(all_chunks)}")
    vectors = embed_texts([c["text"] for c in all_chunks])
    payloads = [{"text": c["text"], "source": c["source"]} for c in all_chunks]
    points = [PointStruct(id=c["id"], vector=v, payload=p) for c, v, p in zip(all_chunks, vectors, payloads)]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print("✅ Embedding sukses & dimasukkan ke Qdrant")

if __name__ == "__main__":
    main()
