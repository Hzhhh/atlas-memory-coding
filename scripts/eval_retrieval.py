# -*- coding: utf-8 -*-
"""SWEContextBench 检索质量评测:用 Relationship 真值测 Add/Search 召回。

模拟官方评测口径:把 Experience 任务当作工程历史 Add 进记忆,
用 Related 任务的 problem_statement 当 Search 查询,
金标 = Relationship 表里链接的那个 Experience 任务的所有块。

用法:
  python scripts/eval_retrieval.py [--lite] [--top-k 100] [--repo django/django]
输出: HitRate / Recall@k / MRR / 时延。
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.store import CodingMemoryStore  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "SWEContextBench", "data")
KS = (5, 10, 25, 50, 100)


def build_messages(exp_row: pd.Series) -> list[dict]:
    """把一个 Experience 任务转成'工程历史'消息对(issue + 修复),模拟平台喂给 Add 的形态。"""
    ps = str(exp_row.get("problem_statement") or "")[:4000]
    patch = str(exp_row.get("patch") or "")[:4000]
    ts = pd.to_datetime(exp_row.get("created_at")).timestamp() if exp_row.get("created_at") else None
    msgs = [{"role": "user", "content": f"[issue] {ps}", "timestamp": int(ts) if ts else None}]
    if patch.strip():
        msgs.append({"role": "assistant", "content": f"[fix patch]\n{patch}", "timestamp": int(ts) if ts else None})
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite", action="store_true", help="用 Lite 子集(300 经验/99 相关)")
    ap.add_argument("--repo", default=None, help="只评一个仓库,如 django/django")
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    exp = pd.read_parquet(os.path.join(DATA, "SWEContextBench_Lite_Experience.parquet" if args.lite else "SWEContextBench_Experience.parquet"))
    rel = pd.read_parquet(os.path.join(DATA, "SWEContextBench_Related_Lite.parquet" if args.lite else "SWEContextBench_Related.parquet"))
    ship = pd.read_parquet(os.path.join(DATA, "SWEContextBench_Relationship.parquet"))

    exp_ids = set(exp["instance_id"])
    rel_ids = set(rel["instance_id"])
    links = ship[ship["related_instance_id"].isin(rel_ids) & ship["experience_instance_id"].isin(exp_ids)]
    print(f"experience={len(exp)} related={len(rel)} usable_links={len(links)}")

    if args.repo:
        exp = exp[exp["repo"] == args.repo]
        links = links[links["related_instance_id"].str.startswith(args.repo.replace("/", "__") + "-")]
        if links.empty:
            sys.exit(f"no links for repo {args.repo}")

    # 按 user_id=repo 隔离,与官方'user_id 唯一隔离边界'一致
    repos = sorted(set(links["related_instance_id"].str.rsplit("-", n=1).str[0].str.replace("__", "/")))
    with tempfile.TemporaryDirectory() as td:
        store = CodingMemoryStore(os.path.join(td, "eval.db"))
        t0 = time.time()
        for repo, grp in exp[exp["repo"].isin(repos)].groupby("repo"):
            for _, row in grp.iterrows():
                store.add(repo, str(row["instance_id"]), build_messages(row), request_id=str(row["instance_id"]))
        add_time = time.time() - t0

        # 金标块 id 前缀: instance_id 作为 session_id 写入
        rank_hits: list[int | None] = []
        search_time = 0.0
        for _, link in links.iterrows():
            related_id, gold_exp = link["related_instance_id"], link["experience_instance_id"]
            repo = related_id.rsplit("-", 1)[0].replace("__", "/")
            qrow = rel[rel["instance_id"] == related_id]
            if qrow.empty:
                continue
            query = str(qrow.iloc[0]["problem_statement"])[:6000]
            t1 = time.time()
            res = store.search(repo, query, top_k=args.top_k)
            search_time += time.time() - t1
            gold_prefix = f"{gold_exp}:"
            hit = next((i for i, r in enumerate(res) if r["id"].startswith(gold_prefix)), None)
            rank_hits.append(hit)

        n = len(rank_hits)
        found = [h for h in rank_hits if h is not None]
        print(f"\nqueries={n}  add_time={add_time:.1f}s  search_time={search_time:.1f}s  avg_search={search_time/max(n,1)*1000:.0f}ms")
        print(f"HitRate@{args.top_k}: {len(found)/n:.4f} ({len(found)}/{n})")
        for k in KS:
            if k <= args.top_k:
                print(f"Recall@{k:3d}: {sum(1 for h in found if h < k)/n:.4f}")
        if found:
            print(f"MRR: {sum(1.0/(h+1) for h in found)/n:.4f}")


if __name__ == "__main__":
    main()
