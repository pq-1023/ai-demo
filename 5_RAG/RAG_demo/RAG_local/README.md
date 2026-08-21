# 私有化企业知识库问答系统（RAG）
## 📖项目简介
本项目是基于**检索增强生成(RAG)** 实现的私有化知识库问答系统，可以导入PDF、TXT文档构建私有知识库，基于文档内容精准回答用户问题。
整体采用模块化解耦开发，支持**本地/云端双Embedding**、**本地Ollama /豆包云端双大模型**一键切换，针对大文档、网络异常、文件错误等场景做容错优化。

>技术栈：Python + Chroma向量数据库 + Sentence‑Transformers + Ollama / 火山方舟豆包API

## ✨功能特性
1. **文档解析**：支持PDF、TXT格式文档导入
2. **文本分片**：段落+句子智能分片策略，保证语义完整性
3. **双嵌入模式**
    - 本地：`all‑MiniLM‑L6‑v2`离线向量模型，数据完全私有化
    - 云端：豆包Embedding接口
4. **向量存储**：Chroma持久化向量数据库
5. **检索流水线**：向量召回 + Rerank重排序 + 相似度阈值过滤
6. **Rerank重排序**（可选）：CrossEncoder模型精排，显著提升检索精度
7. **双大模型切换**
    - Ollama本地部署大模型，完全离线可用
    - 豆包云端API，无需本地显卡
8. **多轮对话**：基于Redis+MySQL冷热存储，支持会话历史、追问改写、历史压缩
9. **流式输出**：打字机效果逐字返回答案
10. **健壮性优化**
    - 全链路异常捕获：文件读取、向量库、网络请求
    - 超大文档分批入库，防止内存溢出
    - 空检索兜底：检索不到内容时返回友好提示
    - 可配置参数，所有变量统一放在`.env`

## 📂项目目录
```
RAG_local
├─ chroma_db                 # Chroma向量数据库持久化目录(自动生成)
├─ model                     #存放本地模型(Embedding、Rerank)
│   ├─ all-MiniLM-L6-v2      # Embedding模型
│   └─ bge-reranker-base     # Rerank重排序模型(可选)
├─ docs                      #放置需要导入的PDF/TXT知识库文件
├─ .env                      #全局配置文件
├─ config.py                 #配置中心，统一读取.env
├─ embedding_provider.py     #向量嵌入层，工厂模式本地/云端
├─ document_loader.py        #文档读取、文本分片
├─ vector_store.py           #向量数据库封装类
├─ rerank_provider.py        #Rerank重排序封装
├─ rag_pipeline.py           #检索流水线、构造RAG提示词、问题改写
├─ llm_provider.py           #大模型统一入口 Ollama/豆包云端
├─ conversation.py           #会话管理器(Redis+MySQL冷热存储)
├─ redis_client.py           #Redis缓存层封装
├─ mysql_client.py           #MySQL持久层封装
├─ main.py                   #控制台程序入口
└─ README.md                 #项目说明文档
```

## 🔧环境依赖
### 1.Python版本
Python >=3.9

### 2.基础依赖
```bash
pip install chromadb sentence-transformers pypdf python-dotenv requests -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.多轮对话依赖（Redis+MySQL）
```bash
pip install redis pymysql -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4.确保Redis和MySQL服务已启动

## 📝配置文件 .env说明
```env
# ========== Embedding向量模型配置 ==========
# local 使用本地模型，cloud 使用豆包云端向量接口
EMBED_MODE=local
LOCAL_EMBED_MODEL_PATH=../model/all-MiniLM-L6-v2

#云端Embedding(EMBED_MODE=cloud时填写)
DOUBAO_API_KEY=
DOUBAO_EMB_URL=https://ark.cn-beijing.volces.com/api/v3/embeddings

# ========== LLM大模型配置 ==========
# ollama=本地大模型，doubao=豆包云端API
LLM_MODE=ollama

# Ollama本地配置
OLLAMA_MODEL=qwen2.5:3b-instruct-q4_K_M
OLLAMA_URL=http://127.0.0.1:11434/api/chat

#豆包云端大模型配置(LLM_MODE=doubao时填写)
DOUBAO_LLM_API_KEY=
DOUBAO_LLM_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
DOUBAO_LLM_MODEL=doubao-pro

# ==========向量库路径 ==========
CHROMA_PATH=../chroma_db

# ==========RAG检索参数 ==========
TOP_K=3
SIMILAR_THRESHOLD=0.20

# ==========Rerank重排序配置(可选)==========
ENABLE_RERANK=False
RERANK_MODEL_PATH=../model/bge-reranker-base
RERANK_HF_MODEL_NAME=BAAI/bge-reranker-base
RERANK_TOP_K=20
RERANK_TOP_N=3

CHUNK_SIZE=400
CHUNK_OVERLAP=150

#分批入库每批次分片数量，防止大文档内存溢出
BATCH_SIZE=20

# ==========MySQL配置(多轮对话)==========
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=123456
MYSQL_DATABASE=rag_chat
MYSQL_CHARSET=utf8mb4

# ==========Redis配置(多轮对话)==========
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# ==========会话管理配置 ==========
MAX_HISTORY_TOKENS=2000
MAX_RECENT_ROUNDS=10
SESSION_TIMEOUT_HOURS=168
ASYNC_SAVE_INTERVAL=5
ENABLE_QUERY_REWRITE=True
DEFAULT_USER=default_user
```
### 参数详解
|参数|说明|
|---|---|
|EMBED_MODE|向量模型模式 `local` /`cloud`|
|LOCAL_EMBED_MODEL_PATH|本地Embedding模型存放路径|
|LLM_MODE|大模型模式 `ollama` /`doubao`|
|OLLAMA_MODEL|Ollama已经下载好的模型名称|
|TOP_K|向量召回片段条数|
|SIMILAR_THRESHOLD|相似度阈值，低于该值直接过滤|
|ENABLE_RERANK|是否开启Rerank重排序|
|RERANK_MODEL_PATH|Rerank本地模型存放路径|
|RERANK_HF_MODEL_NAME|HuggingFace模型名(首次运行自动下载)|
|RERANK_TOP_K|Rerank前向量召回数量(默认20)|
|RERANK_TOP_N|Rerank后保留数量(默认3)|
|CHUNK_SIZE|单个分片最大字符长度|
|CHUNK_OVERLAP|分片重叠字符数，避免上下文被切断|
|BATCH_SIZE|分批入库，一次提交多少个分片生成向量|
|MYSQL_*|MySQL连接配置(多轮对话持久化)|
|REDIS_*|Redis连接配置(多轮对话缓存)|
|MAX_HISTORY_TOKENS|历史对话最大Token数，超过自动压缩|
|ENABLE_QUERY_REWRITE|是否启用追问改写(多轮对话)|

## 🚀运行前准备
### 1.基础环境配置
#### 方式A：使用本地模式（推荐，不需要API‑KEY）
1.下载`all‑MiniLM‑L6‑v2`模型，放到`./model/all-MiniLM-L6-v2`文件夹
2.Ollama安装，执行命令拉取模型
```bash
ollama pull qwen2.5:3b-instruct-q4_K_M
```
3.启动Ollama服务，保持后台运行
4.`.env`配置：`EMBED_MODE=local`、`LLM_MODE=ollama`

#### 方式B：使用云端API模式
1.申请火山方舟豆包API密钥
2.`.env`填入`DOUBAO_API_KEY`、`DOUBAO_LLM_API_KEY`
3.切换配置
```env
EMBED_MODE=cloud
LLM_MODE=doubao
```

### 2.Rerank模型配置（可选）
#### 方式A：自动下载
首次开启 `ENABLE_RERANK=True` 运行时，会自动从 HuggingFace 下载 `BAAI/bge-reranker-base` 模型（约100MB）

#### 方式B：手动下载
从以下任一网站下载模型，放到 `./model/bge-reranker-base/`：
- HuggingFace: https://huggingface.co/BAAI/bge-reranker-base
- ModelScope: https://www.modelscope.cn/models/BAAI/bge-reranker-base

### 3.多轮对话配置（可选）
1.确保 Redis 服务已启动（默认端口6379）
2.确保 MySQL 服务已启动（默认端口3306）
3.创建数据库：
```sql
CREATE DATABASE IF NOT EXISTS rag_chat CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```
4.修改 `.env` 中的 `MYSQL_PASSWORD` 为你的 MySQL 密码

## 🎮启动运行
```bash
python main.py
```
程序启动后控制台菜单
```
====私有化企业知识库问答系统====
1、导入文档  2、知识库问答
请输入功能序号：
```

### 选项1：导入文档
输入文件路径，例如 `./docs/demo.pdf`，程序自动解析、分片、分批入库

### 选项2：知识库问答
输入问题，系统检索知识库，流式返回答案

#### 多轮对话命令
| 命令 | 说明 |
|------|------|
| `exit` / `quit` | 退出程序 |
| `clear` | 清空当前会话历史 |
| `new` | 创建新会话 |
| `sessions` / `list` | 查看历史会话列表 |
| `switch <id>` | 切换会话（输入会话ID前12位即可）|
| `delete <id>` | 删除会话 |
| `archive <id>` | 归档会话（从Redis清除，MySQL保留）|
| `help` | 显示所有命令 |

#### 多轮对话示例
```
你的提问：年假几天        ← 第1轮
年假5天。

你的提问：那病假呢       ← 第2轮（自动改写为"病假的规定"进行检索）
病假需要提供医院开具的诊断证明...

你的提问：工资怎么算     ← 第3轮（自动改写为"病假期间薪资如何计算"）
病假期间薪资按照公司薪酬制度执行。

你的提问：sessions       ← 查看所有会话
====历史会话列表====
  1. [活跃] 新对话 (ID: a1b2c3d4e5f6...)
```

## 📌模块设计说明
### 1.embedding_provider.py
抽象基类定义统一`encode()`接口，工厂函数根据配置自动选择本地模型或云端Embedding接口，增加异常捕获。

### 2.document_loader.py
PDF、TXT文档读取；段落+句子智能分片；文件读取异常、编码异常捕获。

### 3.vector_store.py
封装Chroma数据库初始化、文档入库、向量检索；支持upsert（存在则更新）；数据库异常捕获。

### 4.rerank_provider.py
Rerank重排序封装，基于CrossEncoder模型；支持自动下载HuggingFace模型；精排显著提升检索精度。

### 5.rag_pipeline.py
执行问题改写、向量召回、可选Rerank重排序、相似度过滤、构造含历史的RAG提示词；内置空检索兜底逻辑。

### 6.llm_provider.py
统一大模型调用入口，封装Ollama、豆包API两种流式调用；支持同步调用（用于问题改写）；网络超时、连接失败异常捕获。

### 7.conversation.py
会话管理器核心，实现Redis+MySQL冷热存储；支持异步批量写入MySQL；历史Token压缩（滑动窗口+LLM摘要）。

### 8.redis_client.py
Redis缓存层封装，存储活跃会话的热数据；支持会话Hash、历史List、过期管理（TTL）。

### 9.mysql_client.py
MySQL持久层封装，存储所有会话的冷数据；支持会话/消息的增删改查；自动建表。

### 10.main.py
控制台交互入口，接入ConversationManager；支持多轮对话命令（新会话、切换、归档等）；顶层异常捕获。

## 🛡️异常与容错处理
1. 文件异常：文件不存在、PDF损坏、txt编码错误捕获
2. 向量库异常：数据库初始化失败、入库失败、向量查询失败
3. Embedding异常：本地模型加载失败、云端接口调用失败
4. LLM网络异常：Ollama未启动、请求超时、云端API密钥错误
5. Rerank异常：模型未下载时自动跳过，回退到普通检索
6. 内存优化：超大文档分片分批入库，不会一次性加载全部向量
7. 检索兜底：知识库无匹配内容，模型不会强行编造答案
8. 多轮对话异常：Redis/MySQL连接失败时自动降级为内存存储

## ❓常见问题
### Q1 报错无法连接Ollama
A：确认Ollama服务已经启动，端口11434，`.env`里面OLLAMA_URL地址正确

### Q2 本地模型加载失败 local_files_only
A：模型文件夹路径配置正确，模型完整下载，目录名称与`.env`路径完全一致

### Q3 文档导入内存占用高
A：调小`.env`里面`BATCH_SIZE`，降低单次入库分片数量

### Q4 Rerank模型下载失败
A：网络问题导致HuggingFace超时，请手动从ModelScope下载：https://www.modelscope.cn/models/BAAI/bge-reranker-base

### Q5 多轮对话报错Redis连接失败
A：确认Redis服务已启动：`redis-cli ping` 返回PONG；检查`.env`中Redis配置

### Q6 多轮对话报错MySQL连接失败
A：确认MySQL服务已启动；检查`.env`中MYSQL_PASSWORD是否正确；确认数据库`rag_chat`已创建

### Q7 追问改写后检索结果更差
A：可以关闭追问改写：`.env`中设置`ENABLE_QUERY_REWRITE=False`

## 👨‍💻项目简历亮点
采用模块化架构开发私有化RAG知识库问答系统，项目拆分为文档解析、向量存储、Rerank重排序、检索流水线、LLM调用、多轮对话等模块。

**核心能力**：
- 文档解析：PDF/TXT文档解析，段落+句子智能分片
- 向量存储：Chroma持久化，支持upsert重复导入
- 双Embedding模式：本地sentence-transformers / 云端豆包API一键切换
- Rerank重排序：CrossEncoder精排，检索精度显著提升
- 多轮对话：Redis+MySQL冷热存储，追问改写+历史压缩
- 双LLM模式：Ollama本地大模型 / 豆包云端API
- 流式输出：打字机效果，支持同步/异步调用
- 健壮性：全链路异常捕获、空检索兜底、降级策略