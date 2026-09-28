# -*- coding: utf-8 -*-
"""面向工程文本的代码感知分词器。

编程赛道的 Add/Search 载荷是工程历史(issue 描述、patch、报错栈、会话记录),
通用 NLP 分词会把 ``fix_oauth_redirect_bug`` 切没、把 ``FileNotFoundError`` 当一个词。
要点:
- 先在原始大小写上切 camelCase/snake_case,最后才小写化(否则标识符拆不开);
- 文件路径整体保真 + 拆碎片;
- 错误类型名(原样 + 拆分)双写;
- 轻量词干(plural/ed/ing)双写,解决 loader≠loaders 这类召回损失。
"""
from __future__ import annotations

import re

# 最小停用词:工程语境里 "error/fix/commit" 本身有检索价值,只去掉纯功能词
_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "it", "this", "that", "with", "as", "at", "by",
    "from", "we", "you", "i", "if", "then", "than", "so", "but", "not", "no",
    "do", "does", "did", "have", "has", "had", "will", "would", "can", "could",
    "should", "may", "might", "shall", "there", "here", "when", "what", "which",
    "who", "how", "why", "all", "any", "some", "into", "out", "up", "down",
}

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_PATH_RE = re.compile(r"(?:[\w.\-]+/)+[\w.\-]+")          # requests/api.py, django/db/models
_ERR_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:Error|Exception|Warning)\b")
_NUM_RE = re.compile(r"\b\d+\b")

_SUFFIXES = ("ing", "ed", "es", "s")


def tokenize(text: str) -> list[str]:
    """返回 token 列表;路径、错误名、标识符碎片与轻量词干都写入,保证召回。"""
    if not text:
        return []
    tokens: list[str] = []

    # 1) 文件路径整体先抽走(避免被普通分词打散),再拆出目录/文件名/扩展名碎片
    for p in _PATH_RE.findall(text):
        for part in p.split("/"):
            tokens.extend(_split_ident(part))

    # 2) 错误类型名:原样(小写) + 拆分碎片,双写提高精确与召回
    for e in _ERR_RE.findall(text):
        tokens.append(e.lower())
        tokens.extend(_split_ident(e))

    # 3) 常规标识符与数字(在原始大小写上切,再统一小写)
    for m in _IDENT_RE.finditer(text):
        tokens.extend(_split_ident(m.group(0)))
    tokens.extend(m.group(0) for m in _NUM_RE.finditer(text))

    out: list[str] = []
    for t in tokens:
        t = t.lower()
        if not t or t in _STOP or len(t) < 2:
            continue
        out.append(t)
        stem = _light_stem(t)
        if stem != t:
            out.append(stem)
    return out


def _split_ident(ident: str) -> list[str]:
    """snake_case / kebab-case / camelCase / 混合标识符拆分(输入保持原始大小写)。"""
    parts: list[str] = []
    for seg in re.split(r"[_\-.]+", ident):
        if not seg:
            continue
        parts.append(seg)
        parts.extend(s for s in _CAMEL_RE.split(seg) if s)
    return [p.lower() for p in parts if len(p) > 1]


def _light_stem(t: str) -> str:
    """仅对足够长的词剥掉常见后缀;loaders→loader, passed→pass, loading→load。"""
    for suf in _SUFFIXES:
        if len(t) - len(suf) >= 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t
