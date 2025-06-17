from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str = Field(..., description="用户问题")
    n_results: int = Field(5, description="返回结果数量")
    score_threshold: float = Field(0.7, description="相似度阈值")
    filters: Optional[Dict] = Field(None, description="过滤条件")

class SearchResult(BaseModel):
    """搜索结果模型"""
    content: str
    score: float
    source: str
    page: Optional[int] = None

class QueryResponse(BaseModel):
    """查询响应模型"""
    question: str
    answers: List[SearchResult]
    total_results: int
    processing_time: float

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    message: str
    vector_store_stats: Dict

class BuildStatusResponse(BaseModel):
    """构建状态响应"""
    status: str
    processed_files: int
    total_chunks: int
    message: str