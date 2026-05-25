"""
rag.py — full LangChain RAG pipeline
  - document ingestion (load, split, dedup, embed, upsert)
  - retrieval (vectorstore, retriever, scroll)
  - prompt injection guard
  - all LangChain prompt templates
"""
import os, re, time, uuid
from typing import Optional

from fastapi import HTTPException
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.vectorstores import Qdrant as LangchainQdrant
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

from config import (
    llm, embeddings, qdrant_raw, text_splitter,
    COLLECTION_NAME, BATCH_SIZE, MAX_RETRIES,
    sessions, get_session_memory,
)

# =============================================================================
# Prompt injection guard
# =============================================================================

_INJECTION_PATTERNS = [
    r"ignore (all |previous |above )?instructions",
    r"disregard (all |previous |above )?instructions",
    r"you are now", r"act as (a |an )?",
    r"forget (everything|all|your instructions)",
    r"new instructions", r"system prompt",
    r"jailbreak", r"do anything now", r"dan mode",
]

def check_prompt_injection(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in _INJECTION_PATTERNS)

# =============================================================================
# Prompt templates
# =============================================================================

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a knowledgeable and helpful study assistant.
Answer the student's question ONLY using the provided context.
If the answer is not in the context, say "I couldn't find this in your uploaded material."
Do NOT fabricate information.

Context:
{context}

Question: {question}

Answer:""",
)

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context", "focus", "length_instruction"],
    template="""You are a study assistant creating structured summaries.
{focus}
{length_instruction}
Use **bold** for all section headings and important terms.
Use bullet points (+ prefix) for key facts under each heading.

Material:
{context}

Summary:""",
)

FLASHCARD_PROMPT = PromptTemplate(
    input_variables=["context", "num_cards"],
    template="""You are a study assistant. Generate exactly {num_cards} flashcards from the material below.
Return ONLY a valid JSON array — no extra text, no markdown fences.
[
  {{"question": "...", "answer": "..."}},
  ...
]

Material:
{context}""",
)

MCQ_PROMPT = PromptTemplate(
    input_variables=["context", "num_questions"],
    template="""You are a study assistant. Generate exactly {num_questions} multiple-choice questions.
Return ONLY a valid JSON array — no extra text, no markdown fences.
[
  {{
    "question": "...",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "correct": "A",
    "explanation": "..."
  }}
]

Material:
{context}""",
)

REVISION_PROMPT = PromptTemplate(
    input_variables=["context", "focus"],
    template="""You are a study assistant creating revision notes.
{focus}
Structure with:
1. Key Concepts (bullet points)
2. Important Definitions
3. Formulas / Rules (if applicable)
4. Quick Recap (3-5 sentences)

Material:
{context}

Revision Notes:""",
)

EXAM_PREP_PROMPT = PromptTemplate(
    input_variables=["context", "focus", "difficulty"],
    template="""You are an expert exam coach. Create a personalized exam preparation guide.
Difficulty: {difficulty}
{focus}

Include:
1. Likely exam topics (ranked by importance)
2. Key points to memorize
3. Common mistakes to avoid
4. 3 practice questions with answers
5. Final exam tips

Material:
{context}

Exam Preparation Guide:""",
)

# =============================================================================
# Vectorstore + retriever
# =============================================================================

def _get_vectorstore() -> LangchainQdrant:
    return LangchainQdrant(
        client=qdrant_raw,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )

def get_retriever(session_id: str, tenant_id: str, department: Optional[str], k: int = 8):
    must = [
        FieldCondition(key="metadata.session_id", match=MatchValue(value=session_id)),
        FieldCondition(key="metadata.tenant_id",  match=MatchValue(value=tenant_id)),
    ]
    if department:
        must.append(FieldCondition(key="metadata.department", match=MatchValue(value=department)))

    return _get_vectorstore().as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": 0.15, "filter": Filter(must=must)},
    )

def retrieve_chunks_text(
    session_id: str, tenant_id: str, department: Optional[str],
    query: str, k: int = 8
) -> tuple[str, int]:
    docs = get_retriever(session_id, tenant_id, department, k=k).get_relevant_documents(query)
    if not docs:
        return "", 0
    return "\n\n".join(d.page_content for d in docs), len(docs)

def scroll_chunks(
    session_id: str, tenant_id: str, department: Optional[str], limit: int = 12
) -> list[str]:
    must = [
        FieldCondition(key="metadata.session_id", match=MatchValue(value=session_id)),
        FieldCondition(key="metadata.tenant_id",  match=MatchValue(value=tenant_id)),
    ]
    if department:
        must.append(FieldCondition(key="metadata.department", match=MatchValue(value=department)))
    results, _ = qdrant_raw.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=must),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [p.payload.get("page_content", "") for p in results if p.payload.get("page_content")]

# =============================================================================
# Ingestion pipeline
# =============================================================================

def _deduplicate_docs(docs: list[Document], threshold: float = 0.85) -> list[Document]:
    unique, seen_sets = [], []
    for doc in docs:
        words = set(doc.page_content.lower().split())
        if not words:
            continue
        is_dup = any(
            len(words & s) / len(words | s) >= threshold
            for s in seen_sets if len(words | s) > 0
        )
        if not is_dup:
            unique.append(doc)
            seen_sets.append(words)
    return unique

def _embed_with_retry(texts: list[str]) -> list[list[float]]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return embeddings.embed_documents(texts)
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"[embed] attempt {attempt} failed ({e}), retrying in {wait}s...")
            time.sleep(wait)

def ingest_documents(
    docs: list[Document],
    session_id: str, tenant_id: str, department: str, filename: str,
) -> int:
    chunks = text_splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata.update({
            "session_id": session_id, "tenant_id": tenant_id,
            "department": department, "filename": filename,
        })
    unique = _deduplicate_docs(chunks)
    if not unique:
        return 0

    total_batches = -(-len(unique) // BATCH_SIZE)
    total_stored  = 0

    for i in range(0, len(unique), BATCH_SIZE):
        batch   = unique[i : i + BATCH_SIZE]
        vectors = _embed_with_retry([d.page_content for d in batch])
        points  = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[j],
                payload={"page_content": batch[j].page_content, "metadata": batch[j].metadata},
            )
            for j in range(len(batch))
        ]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                qdrant_raw.upsert(collection_name=COLLECTION_NAME, points=points)
                total_stored += len(points)
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise HTTPException(status_code=502, detail=f"Qdrant upsert failed: {e}")
                time.sleep(2 ** attempt)
        print(f"[upload] batch {i//BATCH_SIZE + 1}/{total_batches} stored ({len(batch)} chunks)")

    return total_stored

def load_file(content: bytes, filename: str) -> list[Document]:
    """Load PDF or TXT bytes into LangChain Documents."""
    tmp = f"/tmp/{uuid.uuid4()}_{filename}"
    with open(tmp, "wb") as f:
        f.write(content)
    try:
        if filename.lower().endswith(".pdf"):
            docs = PyMuPDFLoader(tmp).load()
        elif filename.lower().endswith(".txt"):
            docs = TextLoader(tmp, encoding="utf-8").load()
        else:
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
    finally:
        os.remove(tmp)
    return docs

# =============================================================================
# Chain builders (called from routes)
# =============================================================================

def build_qa_chain(session_id: str, tenant_id: str, dept: Optional[str]):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=get_retriever(session_id, tenant_id, dept, k=8),
        memory=get_session_memory(session_id),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        output_key="answer",
        verbose=False,
    )

def run_llm_chain(prompt: PromptTemplate, **kwargs) -> str:
    return LLMChain(llm=llm, prompt=prompt).run(**kwargs)