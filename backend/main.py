"""
main.py — FastAPI routes only
All RAG logic lives in rag.py
All config/clients live in config.py
"""
import json, uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    sessions, _save_sessions, auto_recover_session,
    qdrant_raw, COLLECTION_NAME, text_splitter,
)
from rag import (
    check_prompt_injection,
    ingest_documents, load_file,
    retrieve_chunks_text, scroll_chunks,
    build_qa_chain, run_llm_chain,
    SUMMARY_PROMPT, FLASHCARD_PROMPT, MCQ_PROMPT,
    REVISION_PROMPT, EXAM_PREP_PROMPT,
)
from qdrant_client.models import Filter, FieldCondition, MatchValue

# =============================================================================
# App
# =============================================================================

app = FastAPI(title="AI Study Assistant", version="3.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Request models
# =============================================================================

class SessionCreateRequest(BaseModel):
    tenant_id:  str = "default"
    department: str = "general"

class QuestionRequest(BaseModel):
    session_id: str
    question:   str
    department: Optional[str] = None

class SummaryRequest(BaseModel):
    session_id: str
    topic:      Optional[str] = None
    department: Optional[str] = None
    length:     str = "medium"

class FlashcardRequest(BaseModel):
    session_id: str
    num_cards:  int = 10
    department: Optional[str] = None

class MCQRequest(BaseModel):
    session_id:    str
    num_questions: int = 5
    topic:         Optional[str] = None
    department:    Optional[str] = None

class RevisionRequest(BaseModel):
    session_id: str
    topic:      Optional[str] = None
    department: Optional[str] = None

class ExamPrepRequest(BaseModel):
    session_id: str
    topic:      Optional[str] = None
    difficulty: str = "medium"
    department: Optional[str] = None

# =============================================================================
# Helper — length instruction for summaries
# =============================================================================

_LENGTH_MAP = {
    "short":  "Write approximately 150 words. Be very concise — only the most essential points.",
    "medium": "Write approximately 300 words. Cover all key concepts with brief explanations.",
    "long":   "Write approximately 600 words. Be thorough — cover all concepts, definitions, and details.",
}

# =============================================================================
# Routes
# =============================================================================

@app.get("/")
def root():
    return {"message": "AI Study Assistant is running", "version": "3.2.0"}


@app.get("/health")
def health():
    try:
        cols    = [c.name for c in qdrant_raw.get_collections().collections]
        qdrant_ok = COLLECTION_NAME in cols
    except Exception:
        qdrant_ok = False
    return {"status": "ok", "version": "3.2.0", "qdrant": "connected" if qdrant_ok else "error"}


@app.post("/session/create")
def create_session(
    req: Optional[SessionCreateRequest] = None,
    tenant_id:  str = Query(default="default"),
    department: str = Query(default="general"),
):
    tid  = req.tenant_id  if req else tenant_id
    dept = req.department if req else department
    sid  = str(uuid.uuid4())
    sessions[sid] = {"doc_names": [], "tenant_id": tid, "department": dept}
    _save_sessions()
    return {"session_id": sid, "tenant_id": tid, "department": dept}


@app.get("/session/{session_id}/docs")
def list_documents(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id]
    return {"session_id": session_id, "tenant_id": s["tenant_id"],
            "department": s["department"], "documents": s["doc_names"]}


@app.post("/upload")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session  = sessions[session_id]
    content  = await file.read()
    filename = file.filename or "unknown"

    docs = load_file(content, filename)
    if not docs or not any(d.page_content.strip() for d in docs):
        raise HTTPException(status_code=400, detail="Could not extract text — is this a scanned PDF?")

    raw_count = len(text_splitter.split_documents(docs))
    stored    = ingest_documents(
        docs=docs, session_id=session_id,
        tenant_id=session["tenant_id"], department=session["department"],
        filename=filename,
    )
    session["doc_names"].append(filename)
    _save_sessions()
    return {"message": f"'{filename}' uploaded successfully",
            "raw_chunks": raw_count, "chunks_after_dedup": stored,
            "documents": session["doc_names"]}


@app.post("/ask")
def ask_question(req: QuestionRequest):
    auto_recover_session(req.session_id, req.department)
    if check_prompt_injection(req.question):
        raise HTTPException(status_code=400, detail="Query rejected: potential prompt injection detected")

    session = sessions[req.session_id]
    dept    = req.department or session["department"]
    chain   = build_qa_chain(req.session_id, session["tenant_id"], dept)
    result  = chain({"question": req.question})

    src_docs   = result.get("source_documents", [])
    num_chunks = len(src_docs)
    sources    = list({d.metadata.get("filename", "unknown") for d in src_docs})

    if num_chunks == 0:
        return {"question": req.question, "answer": "No relevant content found. Try uploading more material.",
                "chunks_used": 0, "sources": [], "department": dept}

    note = " (limited context — consider uploading more material)" if num_chunks < 2 else ""
    return {"question": req.question, "answer": result["answer"] + note,
            "chunks_used": num_chunks, "sources": sources, "department": dept}


@app.post("/summarize")
def summarize(req: SummaryRequest):
    auto_recover_session(req.session_id, req.department)
    session = sessions[req.session_id]
    dept    = req.department or session["department"]
    context, num_chunks = retrieve_chunks_text(req.session_id, session["tenant_id"], dept,
                                               req.topic or "main concepts", k=8)
    if num_chunks == 0:
        raise HTTPException(status_code=400, detail="No relevant content found in uploaded documents")

    summary = run_llm_chain(
        SUMMARY_PROMPT,
        context=context,
        focus=f"Focus on: {req.topic}" if req.topic else "Cover all main ideas.",
        length_instruction=_LENGTH_MAP.get(req.length, _LENGTH_MAP["medium"]),
    )
    return {"summary": summary, "chunks_used": num_chunks, "topic": req.topic or "general", "length": req.length}


@app.post("/flashcards")
def generate_flashcards(req: FlashcardRequest):
    auto_recover_session(req.session_id, req.department)
    session = sessions[req.session_id]
    dept    = req.department or session["department"]
    chunks  = scroll_chunks(req.session_id, session["tenant_id"], dept, limit=12)
    if not chunks:
        raise HTTPException(status_code=400, detail="No documents found for this session")

    raw = run_llm_chain(FLASHCARD_PROMPT, context=" ".join(chunks), num_cards=str(req.num_cards))
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        cards = json.loads(raw)
        return {"flashcards": cards, "count": len(cards)}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse flashcards — try again")


@app.post("/mcq")
def generate_mcq(req: MCQRequest):
    auto_recover_session(req.session_id, req.department)
    if req.topic and check_prompt_injection(req.topic):
        raise HTTPException(status_code=400, detail="Topic rejected: potential prompt injection detected")
    session = sessions[req.session_id]
    dept    = req.department or session["department"]
    context, num_chunks = retrieve_chunks_text(req.session_id, session["tenant_id"], dept,
                                               req.topic or "key concepts", k=6)
    if num_chunks == 0:
        raise HTTPException(status_code=400, detail="No relevant content found in uploaded documents")

    raw = run_llm_chain(MCQ_PROMPT, context=context, num_questions=str(req.num_questions))
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        questions = json.loads(raw)
        return {"mcqs": questions, "count": len(questions)}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse MCQs — try again")


@app.post("/revision-notes")
def revision_notes(req: RevisionRequest):
    auto_recover_session(req.session_id, req.department)
    session = sessions[req.session_id]
    dept    = req.department or session["department"]
    context, num_chunks = retrieve_chunks_text(req.session_id, session["tenant_id"], dept,
                                               req.topic or "all key concepts", k=8)
    if num_chunks == 0:
        raise HTTPException(status_code=400, detail="No relevant content found in uploaded documents")

    notes = run_llm_chain(REVISION_PROMPT, context=context,
                          focus=f"Focus on: {req.topic}" if req.topic else "Cover all major topics.")
    return {"revision_notes": notes, "chunks_used": num_chunks, "topic": req.topic or "general"}


@app.post("/exam-prep")
def exam_prep(req: ExamPrepRequest):
    auto_recover_session(req.session_id, req.department)
    if req.difficulty not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="difficulty must be 'easy', 'medium', or 'hard'")
    session = sessions[req.session_id]
    dept    = req.department or session["department"]
    context, num_chunks = retrieve_chunks_text(req.session_id, session["tenant_id"], dept,
                                               req.topic or "exam key topics", k=8)
    if num_chunks == 0:
        raise HTTPException(status_code=400, detail="No relevant content found in uploaded documents")

    guide = run_llm_chain(EXAM_PREP_PROMPT, context=context,
                          focus=f"Focus on: {req.topic}" if req.topic else "Cover all uploaded material.",
                          difficulty=req.difficulty)
    return {"exam_prep_guide": guide, "chunks_used": num_chunks,
            "topic": req.topic or "general", "difficulty": req.difficulty}


@app.delete("/delete-doc")
def delete_document(session_id: str, filename: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    qdrant_raw.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(must=[
            FieldCondition(key="metadata.session_id", match=MatchValue(value=session_id)),
            FieldCondition(key="metadata.tenant_id",  match=MatchValue(value=session["tenant_id"])),
            FieldCondition(key="metadata.filename",   match=MatchValue(value=filename)),
        ]),
    )
    session["doc_names"] = [d for d in session.get("doc_names", []) if d != filename]
    _save_sessions()
    return {"message": f"'{filename}' deleted from knowledge base", "documents": session["doc_names"]}


@app.delete("/session/{session_id}/memory")
def clear_memory(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id].pop("memory", None)
    return {"message": "Conversation memory cleared"}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    qdrant_raw.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(must=[
            FieldCondition(key="metadata.session_id", match=MatchValue(value=session_id)),
            FieldCondition(key="metadata.tenant_id",  match=MatchValue(value=session["tenant_id"])),
        ]),
    )
    del sessions[session_id]
    _save_sessions()
    return {"message": "Session and all associated vectors deleted"}