from pydantic import BaseModel, Field
from typing import List, Tuple, Optional


class QueryRequest(BaseModel):
    """查询请求模型"""
    session_id: Optional[str] = None
    query: str


class RAGRequest(BaseModel):
    """RAG检索请求模型"""
    query: str


class SessionResponse(BaseModel):
    """会话响应模型"""
    session_id: str
    history: List[Tuple[str, str]]


class AgentStep(BaseModel):
    """Agent执行步骤模型"""
    thought: Optional[str] = None
    tool: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None


class AgentResponse(BaseModel):
    """Agent响应模型"""
    response: str
    session_id: str
    steps: Optional[List[AgentStep]] = None


class RouterResponse(BaseModel):
    """Router Graph响应模型"""
    session_id: str
    route: str
    rag_intent: str = "unknown"
    source_hints: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    response: str
    steps: Optional[List[dict]] = None
    error: Optional[str] = None


class RAGResponse(BaseModel):
    """RAG检索响应模型"""
    response: str


class ReorderRequest(BaseModel):
    """重排序请求模型"""
    query: str
    documents: List[str]


class ReorderResponse(BaseModel):
    """重排序响应模型"""
    documents: List[dict]
