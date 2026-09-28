# -*- coding: utf-8 -*-
"""gpt-4o-mini 重排(OpenRouter 路由):RRF 融合结果上的语义精排。

设计原则:
- 规则对齐:开源组预期 Add/Search 使用 gpt-4o-mini,本模块用的正是该模型;
- 永不失败:超时/解析失败/网络异常 -> 返回 None,调用方退回 RRF 顺序;
- 一次调用:listwise(整批候选编号打分),控制成本与延迟;
- 零新增依赖:stdlib urllib。
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request

ENABLED = os.environ.get("RERANK", "0") == "1"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = os.environ.get("RERANK_MODEL", "openai/gpt-4o-mini")
TOP_N = int(os.environ.get("RERANK_TOP_N", "30"))
TIMEOUT = float(os.environ.get("RERANK_TIMEOUT", "25"))
SNIPPET_CHARS = int(os.environ.get("RERANK_SNIPPET_CHARS", "420"))

_URL = "https://openrouter.ai/api/v1/chat/completions"
_SYS = (
    "You are a relevance judge for a coding agent's long-term memory. "
    "Given a new engineering task and numbered memory candidates retrieved from past work, "
    "decide which memories would actually help solve the new task, and in what order. "
    "Judge by actionable relevance: same bug pattern, same API/module, same fix strategy, "
    "same architectural context. Ignore merely topical or keyword overlap. "
    'Respond with ONLY a JSON object: {"order": [candidate numbers, most helpful first]}. '
    "Include only genuinely helpful candidates; return an empty list if none qualify."
)


def rerank(query: str, candidates: list[dict]) -> list[int] | None:
    """candidates: [{'idx': int, 'content': str}, ...] -> 相关候选的 idx 列表(降序)。

    返回 None 表示重排不可用,调用方保持原顺序。
    """
    if not ENABLED or not API_KEY or not candidates:
        return None
    lines = [f"[{c['idx']}] {c['content'][:SNIPPET_CHARS]}" for c in candidates]
    user = (
        f"New task:\n{query[:3000]}\n\nMemory candidates:\n" + "\n---\n".join(lines)
    )
    body = json.dumps(
        {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": 400,
            "messages": [
                {"role": "system", "content": _SYS},
                {"role": "user", "content": user},
            ],
        }
    ).encode()

    for attempt in range(2):
        try:
            req = urllib.request.Request(
                _URL,
                data=body,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                out = json.loads(resp.read().decode())
            text = out["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", text, re.S)
            order = json.loads(m.group(0))["order"] if m else []
            valid = {c["idx"] for c in candidates}
            seq = [i for i in order if i in valid]
            return seq  # 可为空列表 = LLM 认为全不相关(合法结论,不是失败)
        except Exception as e:
            if attempt == 0:
                time.sleep(1.5)
            else:
                print(f"[rerank] unavailable, falling back to RRF order: {e}")
    return None
