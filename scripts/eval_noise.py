# -*- coding: utf-8 -*-
"""噪声条件评测:模拟 CAMBench 的"相关 + 噪声"双条件计分口径。

对每个 Related 任务:
- 相关记忆:gold 所在仓库的 Experience(同仓库天然含大量同主题干扰项)
- 噪声记忆:额外注入 N 个其他仓库的 Experience(跨仓库噪声)
度量:
- gold 命中率/排名(门控不能伤召回)
- 噪声泄漏率:top-k 里跨仓库块占比(门控应压低)
- 空返回率:完全无关查询应返回空(可选场景)

用法: python scripts/eval_noise.py --lite --noise-repos 3
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
from scripts.eval_retrieval import DATA, build_messages  # noqa: E402

KS = (5, 10, 25, 50, 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite", action="store_true")
    ap.add_argument("--noise-repos", type=int, default=3, help="每个 query 注入几个跨仓库噪声经验")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 个链接(0=全部)")
    args = ap.parse_args()

    exp = pd.read_parquet(os.path.join(
        DATA, "SWEContextBench_Lite_Experience.parquet" if args.lite else "SWEContextBench_Experience.parquet"))
    rel = pd.read_parquet(os.path.join(
        DATA, "SWEContextBench_Related_Lite.parquet" if args.lite else "SWEContextBench_Related.parquet"))
    ship = pd.read_parquet(os.path.join(DATA, "SWEContextBench_Relationship.parquet"))
    exp_ids, rel_ids = set(exp["instance_id"]), set(rel["instance_id"])
    links = ship[ship["related_instance_id"].isin(rel_ids) & ship["experience_instance_id"].isin(exp_ids)]
    if args.limit:
        links = links.sample(n=min(args.limit, len(links)), random_state=42)  # 随机抽样,避开表序聚类

    all_repos = sorted(set(exp["repo"]))
    with tempfile.TemporaryDirectory() as td:
        store = CodingMemoryStore(os.path.join(td, "noise.db"))
        # 噪声仓库池:每个仓库整体灌入一个独立 user(隔离边界内共享,模拟同一系统的跨项目历史)
        # 但为了跨仓库混进同一 user 的效果,直接把噪声任务灌进 query 用户的 user_id
        t0 = time.time()
        hit_ranks: list[int | None] = []
        leak_counts: list[int] = []
        ret_counts: list[int] = []
        noise_user = "noise-pool"
        for li, (_, link) in enumerate(links.iterrows()):
            rid, gid = link["related_instance_id"], link["experience_instance_id"]
            repo = rid.rsplit("-", 1)[0].replace("__", "/")
            # 每次换 user_id 隔离,避免上一个 query 的噪声累积 -> 每轮新建临时 user
            user = f"q{li}"
            # 相关侧:gold 经验任务必须入库(关联可跨仓库,只灌 related 所在仓库会漏金标),
            # 外加 related 仓库全部经验作为同仓库干扰项
            grow = exp[exp["instance_id"] == gid]
            if not grow.empty:
                store.add(user, gid, build_messages(grow.iloc[0]), request_id=gid)
            for _, row in exp[exp["repo"] == repo].iterrows():
                if row["instance_id"] != gid:
                    store.add(user, str(row["instance_id"]), build_messages(row), request_id=str(row["instance_id"]))
            gold_repo = repo
            # 噪声侧:抽 N 个其他仓库的经验灌进同一 user
            other = [r for r in all_repos if r != gold_repo]
            step = max(1, len(other) // args.noise_repos)
            noise_repos = [other[(li * step + k * 7) % len(other)] for k in range(args.noise_repos)]
            for nr in noise_repos:
                for _, row in exp[exp["repo"] == nr].head(10).iterrows():  # 每仓库取10条控制规模
                    store.add(user, f"noise-{nr}-{row['instance_id']}", build_messages(row), request_id=f"noise-{nr}-{row['instance_id']}")
            qrow = rel[rel["instance_id"] == rid]
            query = str(qrow.iloc[0]["problem_statement"])[:6000]
            res = store.search(user, query, top_k=args.top_k)
            gold_prefix = f"{user}:{gid}:"
            hit = next((i for i, r in enumerate(res) if r["id"].startswith(gold_prefix)), None)
            hit_ranks.append(hit)
            noise_prefixes = tuple(f"{user}:noise-{nr}-" for nr in noise_repos)
            leak = sum(1 for r in res if r["id"].startswith(noise_prefixes))
            leak_counts.append(leak)
            ret_counts.append(len(res))

        n = len(hit_ranks)
        found = [h for h in hit_ranks if h is not None]
        print(f"\nnoise_repos={args.noise_repos} queries={n} add_time={time.time()-t0:.0f}s")
        print(f"gold HitRate@{args.top_k}: {len(found)/n:.4f}  MRR: {sum(1/(h+1) for h in found)/n:.4f}")
        print(f"avg return size: {sum(ret_counts)/n:.1f}")
        print(f"cross-repo noise leak: avg {sum(leak_counts)/n:.1f} chunks in top-{args.top_k} "
              f"| queries with 0 leak: {sum(1 for c in leak_counts if c==0)/n:.2%}")
        for k in KS:
            if k <= args.top_k:
                print(f"Recall@{k:3d}: {sum(1 for h in found if h < k)/n:.4f}")


if __name__ == "__main__":
    main()
