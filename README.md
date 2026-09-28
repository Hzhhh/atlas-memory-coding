# aml-coding — AML 编程赛道记忆系统 (v0.1 baseline)

面向 [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai) 编程赛道的长期记忆服务,
严格实现官方 **Add / Search / Health** 契约。

## 设计

参考 claude-mem 的"观察式记忆"思路(会话 → 观察 → 分块 → 索引 → 跨会话检索),
针对工程历史载荷( issue 描述 / patch / 报错栈 / 会话记录)做了编程特化:

| 模块 | 说明 |
|---|---|
| `app/tokenizer.py` | 代码感知分词:camelCase/snake_case 拆分、文件路径保真、错误类型名双写、轻量词干 |
| `app/chunker.py` | 消息级分块:段落贪心聚合 ~700 字符,大 patch 按行滑窗不截断 |
| `app/store.py` | SQLite(WAL) 同步持久化 + Lucene 风格 BM25(恒正 IDF) + 种子邻接扩展检索 |
| `app/main.py` | FastAPI 契约端点,`Authorization: Token` 鉴权 |

契约要点:
- **Add 同步**:写入并落盘后才返回 200,`success=true`,原样回显 `request_id`
- **Search 只返回记忆证据**(`content` 原文),不生成答案;按相关性降序;尊重 `top_k`
- **user_id 严格隔离**(表和索引都按 user_id 分域)
- `/health` 免鉴权,供平台探测

## API

```bash
# 健康检查
curl http://<host>:8001/health

# 写入记忆(同步落盘)
curl -X POST http://<host>:8001/add \
  -H "Content-Type: application/json" -H "Authorization: Token <KEY>" \
  -d '{"request_id":"r1","messages":[{"role":"user","content":"..."}],
       "user_id":"u1","session_id":"s1"}'

# 检索记忆证据
curl -X POST http://<host>:8001/search \
  -H "Content-Type: application/json" -H "Authorization: Token <KEY>" \
  -d '{"query":"...","user_id":"u1","top_k":100}'
```

## 部署

```bash
docker build -t aml-coding .
docker run -d --name aml-coding --restart unless-stopped \
  -p 8001:8000 -e AML_API_TOKEN=$(cat token.txt) \
  -v /srv/aml-coding-data:/data aml-coding
```

环境变量:`AML_API_TOKEN`(鉴权,必设)、`AML_DB_PATH`(默认 `/data/memory.db`)、
`SEED_TOP_N`(种子数,默认 20)、`NEIGHBOR_SPAN`(邻接半径,默认 1)。

## 致谢与披露

- 记忆组织思路(观察-分块-检索)受 [claude-mem](https://github.com/thedotmack/claude-mem)(MIT) 启发;
  本仓库为独立实现(未直接复用其代码),检索与分词为针对工程文本的原创设计。
- 评测协议遵循 Agent Memory Leaderboard 官方 Add/Search 契约。

## License

MIT
