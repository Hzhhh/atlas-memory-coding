# -*- coding: utf-8 -*-
"""记忆存储:SQLite 持久化 + 每用户 BM25 索引 + 邻接扩展检索。

契约要点:
- Add 同步落盘(WAL 提交后才算成功);
- user_id 严格隔离——所有表和索引都带 user_id 维度;
- Search 只返回记忆证据(content 原文),不做任何答案生成。
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field

from .chunker import chunk_message
from .tokenizer import tokenize

DB_PATH = os.environ.get("AML_DB_PATH", "/data/memory.db")
SEED_TOP_N = int(os.environ.get("SEED_TOP_N", "20"))     # 种子命中数
NEIGHBOR_SPAN = int(os.environ.get("NEIGHBOR_SPAN", "1"))  # 邻接扩展半径
MAX_RETURN = int(os.environ.get("MAX_RETURN", "100"))


class BM25:
    """Lucene 风格 BM25:idf = ln(1 + (N-df+.5)/(df+.5)) 恒正,

    不像 rank_bm25.BM25Okapi 在小语料/高df词上会出现负 idf 把整表分数打负。"""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_tfs: list[Counter] = [Counter(c) for c in corpus]
        self.doc_len = [len(c) for c in corpus]
        self.avgdl = (sum(self.doc_len) / len(corpus)) if corpus else 0.0
        n = len(corpus)
        df: Counter = Counter()
        for tf in self.doc_tfs:
            df.update(tf.keys())
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        out = [0.0] * len(self.doc_tfs)
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.doc_tfs):
                f = tf.get(term, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    out[i] += idf * f * (self.k1 + 1) / (f + norm)
        return out


@dataclass
class _UserIndex:
    """单用户的内存态 BM25 索引(dirty 时重建)。"""
    chunks: list[dict] = field(default_factory=list)
    bm25: BM25 | None = None
    dirty: bool = True

    def rebuild(self) -> None:
        corpus = [c["tokens"] for c in self.chunks]
        self.bm25 = BM25(corpus) if corpus else None
        self.dirty = False


class CodingMemoryStore:
    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._lock = threading.RLock()
        self._idx: dict[str, _UserIndex] = {}
        self._init_db()

    # ---------- 持久化 ----------
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS chunks(
                       id TEXT PRIMARY KEY,
                       user_id TEXT NOT NULL,
                       session_id TEXT NOT NULL,
                       msg_idx INTEGER NOT NULL,
                       chunk_idx INTEGER NOT NULL,
                       role TEXT,
                       content TEXT NOT NULL,
                       created_at TEXT NOT NULL,
                       tokens_json TEXT NOT NULL)"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user ON chunks(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_sess ON chunks(user_id, session_id)")

    def stats(self) -> dict:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT user_id), COUNT(DISTINCT session_id) FROM chunks"
            ).fetchone()
        return {"chunks": row[0], "users": row[1], "sessions": row[2]}

    # ---------- Add ----------
    def add(self, user_id: str, session_id: str, messages: list[dict]) -> int:
        """同步写入并提交;返回写入块数。失败抛异常 -> 上层 5xx,不会假成功。"""
        rows: list[tuple] = []
        mem_chunks: list[dict] = []
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for mi, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""
            ts = msg.get("timestamp")
            created = (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)) if ts else now
            )
            for ci, piece in enumerate(chunk_message(content)):
                cid = f"{session_id}:{mi}:{ci}:{uuid.uuid4().hex[:8]}"
                toks = tokenize(piece)
                rows.append(
                    (cid, user_id, session_id, mi, ci, role, piece, created,
                     json.dumps(toks))
                )
                mem_chunks.append(
                    {"id": cid, "session_id": session_id, "msg_idx": mi,
                     "chunk_idx": ci, "role": role, "content": piece,
                     "created_at": created, "tokens": toks}
                )
        with self._lock:
            with self._conn() as conn:
                conn.executemany(
                    "INSERT INTO chunks(id,user_id,session_id,msg_idx,chunk_idx,role,content,created_at,tokens_json)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    rows,
                )
            # 增量维护内存索引:索引未装载时 _get_index 会连本次新行一起从磁盘读出,
            # 已装载时才需要手动追加,否则会双重计入
            existed = user_id in self._idx
            self._get_index(user_id)
            if existed:
                self._idx[user_id].chunks.extend(mem_chunks)
            self._idx[user_id].dirty = True
        return len(rows)

    # ---------- Search ----------
    def search(self, user_id: str, query: str, top_k: int = 100) -> list[dict]:
        with self._lock:
            idx = self._get_index(user_id)
            if idx.bm25 is None or not idx.chunks:
                return []
            scores = idx.bm25.get_scores(tokenize(query))
            order = sorted(range(len(idx.chunks)), key=lambda i: scores[i], reverse=True)
            seeds = [i for i in order[:SEED_TOP_N] if scores[i] > 0]

            # 邻接扩展:种子块带上前后邻居,patch 上下文不被切走;
            # 邻居按 0.99 降权,保证种子永远排在它拉进来的邻居之前
            picked: dict[int, float] = {}
            for rank, i in enumerate(seeds):
                for j in range(i - NEIGHBOR_SPAN, i + NEIGHBOR_SPAN + 1):
                    if not 0 <= j < len(idx.chunks):
                        continue
                    cand = scores[i] if j == i else scores[i] * 0.99 - 1e-6 * (rank + 1)
                    picked[j] = max(picked.get(j, 0.0), cand)

            items = []
            for j, s in sorted(picked.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
                c = idx.chunks[j]
                items.append(
                    {
                        "id": c["id"],
                        "content": c["content"],
                        "score": round(float(s), 4),
                        "created_at": c["created_at"],
                    }
                )
            return items

    def _get_index(self, user_id: str) -> _UserIndex:
        idx = self._idx.get(user_id)
        if idx is None:
            idx = _UserIndex()
            with self._conn() as conn:
                for row in conn.execute(
                    "SELECT id,session_id,msg_idx,chunk_idx,role,content,created_at,tokens_json"
                    " FROM chunks WHERE user_id=? ORDER BY session_id,msg_idx,chunk_idx",
                    (user_id,),
                ):
                    idx.chunks.append(
                        {
                            "id": row[0], "session_id": row[1], "msg_idx": row[2],
                            "chunk_idx": row[3], "role": row[4], "content": row[5],
                            "created_at": row[6], "tokens": json.loads(row[7]),
                        }
                    )
            self._idx[user_id] = idx
        if idx.dirty:
            idx.rebuild()
        return idx
