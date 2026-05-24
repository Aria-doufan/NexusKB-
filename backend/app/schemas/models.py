from pydantic import BaseModel, Field
from typing import Any, Dict, List, Tuple, Optional


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
    request_id: Optional[str] = None
    debug_id: Optional[str] = None
    rag_intent: str = "unknown"
    source_hints: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    response: str
    steps: Optional[List[dict]] = None
    error: Optional[str] = None


class MemoryItemResponse(BaseModel):
    """长期记忆条目响应模型"""
    id: str
    user_id: str
    session_id: str
    memory: str
    memory_type: str
    source: str = "chat"
    source_message_ids: List[Any] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    status: str = "active"


class MemoryListResponse(BaseModel):
    """长期记忆列表响应模型"""
    memories: List[MemoryItemResponse] = Field(default_factory=list)


class MemoryListSuccessResponse(BaseModel):
    """长期记忆列表成功响应包裹模型"""
    code: int
    message: str
    data: MemoryListResponse


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
