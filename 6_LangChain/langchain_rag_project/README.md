# LangChain_RAG 知识库问答系统

基于 LangChain 框架构建的本地 RAG（检索增强生成）问答系统，支持文档入库、向量检索、Rerank 重排、多轮对话。

## 目录结构

```
langchain_rag_project/
├── .env                    # 配置文件
├── config.py               # 配置读取
├── embedding_provider.py   # Embedding 模型（all-MiniLM-L6-v2）
├── vector_db.py            # Chroma 向量数据库
├── document_loader.py      # 文档加载（PDF/TXT/MD）
├── text_splitter.py        # 文本切块
├── rag_retriever.py        # 检索器（Rerank 可开关）
├── rerank_module.py        # Rerank 重排序
├── rag_chain.py            # RAG 链编排
├── ingest.py               # 文档入库
├── chat.py                 # 主程序
└── model/                  # 本地模型目录
```

## 环境要求

- Python >= 3.10
- Ollama（本地 LLM 服务）
- 依赖包：

```bash
pip install langchain langchain-chroma langchain-ollama langchain-text-splitters langchain-huggingface langchain-classic pypdf python-dotenv
```

## 配置说明

编辑 `.env` 文件：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `CHUNK_SIZE` | 文本切块大小（字符） | 600 |
| `CHUNK_OVERLAP` | 切块重叠大小 | 120 |
| `CHROMA_PATH` | 向量库存储路径 | ./chroma_db |
| `COLLECTION_NAME` | 集合名称 | knowledge_base |
| `TOP_K` | 检索返回条数 | 4 |
| `EMBED_MODEL_PATH` | Embedding 模型路径 | ./model/all-MiniLM-L6-v2 |
| `ENABLE_RERANK` | 是否启用 Rerank | False |
| `RERANK_MODEL_PATH` | Rerank 模型路径 | ./model/bge-reranker-base |
| `RERANK_TOP_N` | Rerank 后保留条数 | 3 |
| `OLLAMA_MODEL` | Ollama 模型名称 | qwen2.5:3b-instruct-q4_K_M |
| `OLLAMA_URL` | Ollama 服务地址 | http://127.0.0.1:11434 |
| `MAX_HISTORY_ROUNDS` | 多轮对话历史轮数 | 10 |

## 启动流程

### 1. 启动 Ollama

```bash
ollama serve
```

### 2. 拉取模型（首次使用）

```bash
ollama pull qwen2.5:3b-instruct-q4_K_M
```

### 3. 运行主程序

```bash
python chat.py
```

启动后菜单：

```
请选择操作:
  [1] 问答对话
  [2] 读取文件入库
  [0] 退出
```

## 功能说明

### 文件入库

选择 `[2]` 后输入文件路径，支持 PDF、TXT、MD 格式：

```
请输入文件路径: ./test.pdf
入库方式 [1] 追加  [2] 清空重建 (回车默认追加): 2
```

- 追加模式：在现有向量库基础上添加
- 清空重建：删除旧数据重新入库

### 问答对话

选择 `[1]` 进入对话模式，支持以下命令：

| 命令 | 说明 |
|------|------|
| `exit` | 退出程序 |
| `clear` | 清空对话历史 |
| `history` | 查看对话历史 |
| `file` | 随时追加文件入库 |

### Rerank 重排

将 `.env` 中 `ENABLE_RERANK=True` 并下载 Rerank 模型到指定路径：

```bash
# 下载 bge-reranker-base 模型
huggingface-cli download BAAI/bge-reranker-base --local-dir ./model/bge-reranker-base
```

## 工作流程

```
文档 → 加载 → 切块 → Embedding → Chroma存储
                                        ↓
用户问题 → Embedding → 向量检索 → Rerank重排 → LLM生成 → 返回答案
```