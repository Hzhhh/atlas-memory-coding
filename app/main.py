# -*- coding: utf-8 -*-
"""AML 编程赛道记忆服务入口:严格实现官方 Add / Search / Health 契约。

与官方文档一致的字段(与已通过评测的文本赛道服务对齐):
- Add   同步: 写入并持久化完成后才回 200; success=true; 原样返回 request_id
- Search data 按相关性降序; 尊重 top_k; content 直接喂给平台 Answer 模型
- user_id 为唯一隔离边界; 鉴权 Authorization: Token <key>

启动: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .store import CodingMemoryStore

API_TOKEN = os.environ.get("AML_API_TOKEN", "")  # 空=不鉴权(本地调试),上报必须设置

app = FastAPI(title="AML coding memory system", version="0.1.0")
store = CodingMemoryStore()


async def verify_token(authorization: str = Header(default="")) -> None:
    """与官方一致:Authorization: Token <key>;/health 免鉴权供探测。"""
    if not API_TOKEN:
        return
    if authorization != f"Token {API_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


# ---------- 请求/响应模型(对齐官方 schema) ----------
class Message(BaseModel):
    role: str
    content: str
    timestamp: int | None = None


class AddRequest(BaseModel):
    request_id: str
    messages: list[Message]
    user_id: str
    session_id: str


class AddResponse(BaseModel):
    success: bool = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str
    options: list[str] | None = None
    user_id: str
    top_k: int = 100


class MemoryItem(BaseModel):
    id: str
    content: str
    score: float | None = None
    created_at: str | None = None


class SearchResponse(BaseModel):
    data: list[MemoryItem] = Field(default_factory=list)


# ---------- 契约端点 ----------
@app.post("/add", response_model=AddResponse, dependencies=[Depends(verify_token)])
def add(req: AddRequest) -> AddResponse:
    messages = [m.model_dump() for m in req.messages]
    store.add(req.user_id, req.session_id, messages, request_id=req.request_id)
    return AddResponse(request_id=req.request_id, user_id=req.user_id, session_id=req.session_id)


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(verify_token)])
def search(req: SearchRequest) -> SearchResponse:
    items = store.search(req.user_id, req.query, top_k=req.top_k)
    return SearchResponse(data=[MemoryItem(**i) for i in items])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ---------- 自用端点(契约之外,便于自查) ----------
@app.get("/debug/stats")
def stats() -> dict:
    return store.stats()
