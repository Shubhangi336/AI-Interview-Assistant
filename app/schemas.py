"""
Pydantic schemas for all API request and response models.
"""
from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


# ── Common ────────────────────────────────────────────────────────────────────

class SourceDoc(BaseModel):
    source: str
    excerpt: str


class BaseResponse(BaseModel):
    success: bool = True
    message: str = "OK"


# ── User Profile ──────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Candidate's full name")
    role: str = Field(..., min_length=1, max_length=100, description="Target job role/title")
    experience_level: str = Field(
        ...,
        description="Experience level: Fresher, Junior, Mid-Level, Senior, or Lead",
        pattern="^(Fresher|Junior|Mid-Level|Senior|Lead)$",
    )


# ── Question Generation ───────────────────────────────────────────────────────

class QuestionSetRequest(UserProfile):
    background: Optional[str] = Field(
        default="",
        max_length=2000,
        description="Brief background summary (optional)",
    )


class QuestionSetResponse(BaseResponse):
    name: str
    role: str
    experience_level: str
    response: str
    sources: List[SourceDoc] = []


# ── Model Answer ──────────────────────────────────────────────────────────────

class ModelAnswerRequest(UserProfile):
    question: str = Field(..., min_length=5, max_length=1000, description="The interview question to answer")


class ModelAnswerResponse(BaseResponse):
    question: str
    role: str
    experience_level: str
    response: str
    sources: List[SourceDoc] = []


# ── Resume Analysis ───────────────────────────────────────────────────────────

class ResumeAnalysisRequest(UserProfile):
    resume_text: str = Field(
        ...,
        min_length=50,
        max_length=8000,
        description="Paste resume text or job background",
    )


class ResumeAnalysisResponse(BaseResponse):
    name: str
    role: str
    experience_level: str
    response: str
    sources: List[SourceDoc] = []


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(UserProfile):
    message: str = Field(..., min_length=1, max_length=2000)
    history: Optional[List[ChatMessage]] = Field(default_factory=list)


class ChatResponse(BaseResponse):
    user_message: str
    response: str
    sources: List[SourceDoc] = []


# ── Document Ingestion ────────────────────────────────────────────────────────

class IngestResponse(BaseResponse):
    chunks_added: int
    detail: str


# ── Live Mock Answer Evaluation ─────────────────────────────────────────────

class AnswerEvaluationRequest(UserProfile):
    question: str = Field(..., min_length=5, max_length=1500, description="The interview question answered")
    candidate_answer: str = Field(..., min_length=5, max_length=5000, description="The candidate's response")


class AnswerEvaluationResponse(BaseResponse):
    question: str
    candidate_answer: str
    role: str
    experience_level: str
    response: str
    score: Optional[float] = None
    sources: List[SourceDoc] = []


# ── STAR Story Builder & Critique ──────────────────────────────────────────

class StarCritiqueRequest(BaseModel):
    role: str = Field(..., min_length=1, max_length=100)
    experience_level: str = Field(..., pattern="^(Fresher|Junior|Mid-Level|Senior|Lead)$")
    situation: str = Field(..., min_length=10, max_length=2000, description="Context / background")
    task: str = Field(..., min_length=10, max_length=2000, description="Goal / challenge")
    action: str = Field(..., min_length=10, max_length=3000, description="Specific actions taken")
    result: str = Field(..., min_length=10, max_length=2000, description="Outcome / impact / metrics")


class StarCritiqueResponse(BaseResponse):
    role: str
    experience_level: str
    response: str


# ── Resume Parsing ─────────────────────────────────────────────────────────

class ResumeParseResponse(BaseResponse):
    filename: str
    text: str
    character_count: int


# ── Knowledge Base Inspection & Search ─────────────────────────────────────

class KnowledgeDocItem(BaseModel):
    name: str
    type: str
    size_bytes: int
    chunks_count: int


class KnowledgeBaseStatsResponse(BaseResponse):
    total_documents: int
    total_chunks: int
    documents: List[KnowledgeDocItem] = []
    mode: str = "keyword-based retrieval"


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    role: Optional[str] = None
    top_k: Optional[int] = 5


class KnowledgeSearchResponse(BaseResponse):
    query: str
    results: List[SourceDoc] = []


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model: str
    vector_store: str
    is_live_watsonx: bool = False
    version: str = "1.1.0"

