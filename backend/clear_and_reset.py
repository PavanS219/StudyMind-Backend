"""
Run this ONCE to wipe all vectors from Qdrant and clear sessions.json
Then restart the server and re-upload your PDFs fresh.

Usage:
  python clear_and_reset.py
"""
import os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION     = "study_chunks_lc"
EMBED_DIM      = 1536

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False, timeout=60)

# 1. Delete existing collection entirely
existing = [c.name for c in client.get_collections().collections]
if COLLECTION in existing:
    client.delete_collection(COLLECTION)
    print(f"✅ Deleted collection: {COLLECTION}")
else:
    print(f"ℹ️  Collection didn't exist: {COLLECTION}")

# 2. Recreate it fresh
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
)
print(f"✅ Recreated collection: {COLLECTION}")

# 3. Recreate payload indexes
for field in ["metadata.session_id", "metadata.tenant_id", "metadata.department", "metadata.filename"]:
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name=field,
        field_schema=PayloadSchemaType.KEYWORD,
    )
print("✅ Payload indexes created")

# 4. Wipe sessions.json
sf = Path("sessions.json")
if sf.exists():
    sf.write_text("{}")
    print("✅ sessions.json cleared")

print("\n🎉 Done! Now restart the server and re-upload your PDFs.")