FROM python:3.12-slim
WORKDIR /srv/aml-coding
COPY requirements.txt .
# 先装 CPU-only torch(否则默认轮子拖入 ~2GB CUDA 依赖),再装其余依赖
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
COPY app ./app
# 模型不打进镜像:运行时挂载 /models/bge-small-en-v1.5(由 AML_EMBED_MODEL 指定),
# 缺模型时服务自动降级纯 BM25,不会启动失败
ENV AML_DB_PATH=/data/memory.db
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]
