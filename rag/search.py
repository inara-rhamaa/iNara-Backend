import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from google.generativeai import configure, embed_content

load_dotenv()

configure(api_key=os.getenv("GOOGLE_API_KEY"))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "nara_documents")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def search_docs(query: str, top_k: int = 5) -> list[str]:
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
