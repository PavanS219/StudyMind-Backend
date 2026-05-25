"""
config.py — shared config, clients, and session store
All other modules import from here.
"""
import os, json, time, uuid
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import OpenAIEmbeddings as OpenRouterEmbeddings
from langchain_community.vectorstores import Qdrant as LangchainQdrant
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, PayloadSchemaType,
)

# ── Env vars ──────────────────────────────────────────────────────────────────
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_URL         = os.getenv("QDRANT_URL")
QDRANT_API_KEY     = os.getenv("QDRANT_API_KEY")

for val, name in [
    (GROQ_API_KEY,       "GROQ_API_KEY"),
    (OPENROUTER_API_KEY, "OPENROUTER_API_KEY"),
    (QDRANT_URL,         "QDRANT_URL"),
    (QDRANT_API_KEY,     "QDRANT_API_KEY"),
]:
    if not val:
        raise RuntimeError(f"Missing env variable: {name}")

GROQ_MODEL      = "llama-3.3-70b-versatile"
COLLECTION_NAME = "study_chunks_lc"
EMBED_DIM       = 1536
BATCH_SIZE      = 20
MAX_RETRIES     = 3

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    temperature=0.3,
    max_tokens=4096,
)

# ── Embeddings ────────────────────────────────────────────────────────────────
embeddings = OpenRouterEmbeddings(
    model="openai/text-embedding-3-small",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    tiktoken_model_name="text-embedding-3-small",
)

# ── Qdrant client ─────────────────────────────────────────────────────────────
qdrant_raw = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    check_compatibility=False,
    timeout=120,
)

# Bootstrap collection + payload indexes (idempotent — safe on every startup)
existing = [c.name for c in qdrant_raw.get_collections().collections]
if COLLECTION_NAME not in existing:
    qdrant_raw.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

for _field in ["metadata.session_id", "metadata.tenant_id", "metadata.department", "metadata.filename"]:
    try:
        qdrant_raw.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=_field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print(f"[qdrant] index ready: {_field}")
    except Exception as _e:
        print(f"[qdrant] index note ({_field}): {_e}")

# ── Text splitter ─────────────────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ── Session store (persisted to sessions.json) ────────────────────────────────
SESSIONS_FILE = Path("sessions.json")

def _load_sessions() -> dict:
    if SESSIONS_FILE.exists():
        try:
            return json.loads(SESSIONS_FILE.read_text())
        except Exception as e:
            print(f"[sessions] load error: {e} — starting fresh")
    return {}

def _save_sessions() -> None:
    serialisable = {
        sid: {k: v for k, v in s.items() if k != "memory"}
        for sid, s in sessions.items()
    }
    SESSIONS_FILE.write_text(json.dumps(serialisable, indent=2))

sessions: dict[str, dict] = _load_sessions()
print(f"[sessions] Loaded {len(sessions)} session(s) from disk")

def get_session_memory(session_id: str) -> ConversationBufferMemory:
    if "memory" not in sessions[session_id]:
        sessions[session_id]["memory"] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
    return sessions[session_id]["memory"]

def auto_recover_session(session_id: str, department: Optional[str] = None):
    """Restore minimal session metadata if server was restarted."""
    if session_id not in sessions:
        sessions[session_id] = {
            "doc_names":  [],
            "tenant_id":  "default",
            "department": department or "general",
        }
        _save_sessions()
