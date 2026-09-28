# -*- coding: utf-8 -*-
"""工程历史分块:按消息粒度切,超长内容按段落/代码块边界二次切分。

claude-mem 的观察式记忆把会话压成"一条观察一个事实";这里 v0.1 保守一点——
消息内按空行分段、段落聚合到 ~700 字符窗口,保证 patch/栈帧不被拦腰截断。
"""
from __future__ import annotations

MAX_CHARS = 700
OVERLAP_LINES = 2


def chunk_message(content: str, max_chars: int = MAX_CHARS) -> list[str]:
    """把单条消息切成检索友好的块:先按空行分段,再贪心聚合,超长段按行滑窗。"""
    if not content or not content.strip():
        return []
    paragraphs = [p.strip() for p in re_split_blank(content) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for para in paragraphs:
        plen = len(para)
        if plen > max_chars:  # 超长段(大 patch/长栈)单独滑窗
            if buf:
                chunks.append("\n\n".join(buf))
                buf, size = [], 0
            chunks.extend(slide_window(para, max_chars))
            continue
        if size + plen + 2 > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += plen + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def re_split_blank(text: str) -> list[str]:
    return text.replace("\r\n", "\n").split("\n\n")


def slide_window(text: str, max_chars: int) -> list[str]:
    lines = text.split("\n")
    out: list[str] = []
    cur: list[str] = []
    size = 0
    for line in lines:
        llen = len(line) + 1
        if size + llen > max_chars and cur:
            out.append("\n".join(cur))
            cur = cur[-OVERLAP_LINES:]  # 保留少量上下文行,栈帧不脱节
            size = sum(len(l) + 1 for l in cur)
        cur.append(line)
        size += llen
    if cur:
        out.append("\n".join(cur))
    return out
