# 扫地机器人智能客服 Agent

一个面向扫地机器人/扫拖一体机场景的本地化 RAG + Agent 项目。项目使用 LangChain、Ollama、Chroma、MCP、FastAPI、SQLAlchemy 和 Streamlit 构建，支持垂直领域知识库问答、工具调用、实时天气查询、会话持久化、知识库上传重建和个性化报告生成。



## 功能特性

- 垂直领域客服：聚焦扫地机器人选购、使用、维护保养、故障排查、地图建图、基站设置等问题。
- RAG 知识库问答：支持本地文档加载、切分、向量化、混合检索和引用来源展示。
- Agent 工具调用：通过 LangChain Agent 调用天气、用户信息、使用记录、RAG 总结和报告生成工具。
- MCP 天气接入：通过高德地图 MCP 工具获取实时天气，避免把外部 API 逻辑写死在业务工具里。
- FastAPI 服务化：提供聊天、流式聊天、报告、工具和知识库重建接口。
- Streamlit Web UI：提供聊天界面、知识库上传、工具查看、会话查看和检索评估入口。
- 会话持久化：使用 SQLAlchemy 保存会话、消息、工具调用日志和报告结果，默认 SQLite，可切换 MySQL。
- 可选 LoRA/GGUF 模型：可通过 llama.cpp server 接入领域微调模型，用于 RAG 总结和最终回答优化。
- 可选 reranker：支持 BGE reranker 对召回结果重排，默认关闭，避免仓库依赖大模型权重。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| Agent | LangChain `create_agent`、middleware、tool registry |
| LLM | Ollama `qwen3:8b` |
| Embedding | Ollama `qwen3-embedding:0.6b` |
| RAG | Chroma、LangChain Retriever、RecursiveCharacterTextSplitter |
| Rerank | BGE reranker，可选 |
| 外部工具 | MCP、高德地图 Web 服务 |
| API | FastAPI、SSE 风格流式事件 |
| Web UI | Streamlit |
| 数据库 | SQLAlchemy、SQLite / MySQL |
| 可选微调模型 | llama.cpp server + LoRA/GGUF |

## 架构说明

```text
Streamlit Web UI
        |
        | HTTP / SSE
        v
FastAPI API Layer
        |
        v
LangChain Agent
        |
        |-- Local tools: 用户信息、月份、使用记录、报告
        |-- MCP tools: 高德天气
        |-- RAG tool: Chroma 检索 + 知识库总结
        |
        v
Ollama qwen3:8b 负责 Agent 编排和工具调用

可选增强：
llama.cpp LoRA/GGUF 负责 RAG 总结和最终回答表达优化
```

当前推荐架构是：**Ollama qwen3:8b 负责 Agent 主流程和工具调用，LoRA/GGUF 模型只作为最终回答增强层**。这样可以保留工具调用稳定性，同时利用领域微调模型优化客服口吻和领域边界。

## 目录结构

```text
agent/          Agent 主流程、工具、中间件和工具注册中心
api/            FastAPI 服务层
config/         模型、向量库、Prompt 和业务配置
data/           示例知识库和模拟用户使用记录
model/          LLM / Embedding 工厂与 llama.cpp 客户端
prompts/        主提示词、RAG 总结提示词、报告提示词
rag/            文档加载、切分、检索、rerank 和 RAG 总结
scripts/        启动、知识库重建和检索调试脚本
storage/        数据库连接、ORM 模型和仓储方法
utils/          配置、路径、文件、日志和 Prompt 加载工具
app.py          Streamlit 前端入口
start.ps1       FastAPI + Streamlit 一键启动脚本
```

## 环境准备

建议使用 Python 3.10+。

```powershell
pip install -r requirements.txt
```

如果需要启用 BGE reranker，再额外安装：

```powershell
pip install -r requirements-reranker.txt
```

安装并启动 Ollama，然后拉取模型：

```powershell
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b
```

## 环境变量

复制 `.env.example` 为 `.env`：

```powershell
copy .env.example .env
```

配置高德地图 Web 服务 API Key：

```text
AMAP_MAPS_API_KEY=your_amap_web_service_api_key
```

数据库连接可选。默认不配置时，项目会在根目录创建本地 SQLite 文件 `agent_runtime.db`：

```text
DATABASE_URL=mysql+pymysql://root:your_password@127.0.0.1:3306/robot_agent?charset=utf8mb4
```

如果使用 MySQL，需要先创建数据库：

```sql
CREATE DATABASE robot_agent DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

## 启动项目

推荐启动顺序：

```text
Ollama -> 可选 llama.cpp server -> start.ps1
```

启动 FastAPI 和 Streamlit：

```powershell
.\start.ps1
```

启动后访问：

```text
Streamlit: http://127.0.0.1:8501
FastAPI Docs: http://127.0.0.1:8000/docs
```

也可以分别启动：

```powershell
uvicorn api.main:app --reload --port 8000
streamlit run app.py
```

## 可选：接入 LoRA/GGUF 微调模型

仓库不包含 GGUF、LoRA adapter 或训练数据。你可以自行使用 llama.cpp server 启动本地领域模型：

```powershell
cd E:\llama.cpp-b6282\build\bin\Release
.\llama-server.exe `
  -m "E:\llama_models\robot_qwen3_v4_q4km.gguf" `
  -c 4096 `
  -ngl 20 `
  --host 127.0.0.1 `
  --port 8080
```

然后在 `config/rag.yml` 中开启：

```yaml
llama_cpp_enabled: true
llama_cpp_base_url: http://127.0.0.1:8080
```

如果不启用，项目会使用 Ollama 主模型完成 RAG 总结和 Agent 原始回答。

## 知识库管理

项目提供两种知识库维护方式：

1. 在 Streamlit 的“知识库”页面上传文档，然后点击重建知识库。
2. 使用脚本重建或调试检索：

```powershell
py scripts/rebuild_kb.py
py scripts/debug_retrieval.py "扫地机器人怎么保养"
```

支持的文件类型由 `config/chroma.yml` 控制：

```yaml
allow_knowledge_file_type: ["txt", "pdf", "doc", "docx", "xlsx", "xls", "pptx", "csv"]
```

向量库目录 `chroma_db/` 是运行产物，不提交到 GitHub。首次运行或上传文档后需要重建。

## API 接口

FastAPI 默认挂载在 `/api/v1`：

```text
GET  /api/v1/health
GET  /api/v1/tools
POST /api/v1/tools/call
POST /api/v1/chat
POST /api/v1/chat/stream
POST /api/v1/report/generate
POST /api/v1/knowledge/rebuild
GET  /api/v1/conversations
GET  /api/v1/conversations/{conversation_id}/messages
GET  /api/v1/conversations/{conversation_id}/tool-logs
```

普通聊天请求示例：

```json
{
  "query": "扫地机器人怎么保养？",
  "conversation_id": null,
  "include_events": true
}
```

流式接口 `/api/v1/chat/stream` 返回结构化事件：

```text
metadata
tool_call
tool_result
answer
error
done
```

## 后续优化方向

- 增加 Docker Compose，统一启动 Ollama 以外的服务。
- 增加单元测试和 API 测试。
- 增加知识库文档删除、更新、入库状态和切片数量展示。
- 增加检索评估集和评估结果可视化。
- 增加多知识库、多用户和权限控制。
- 增加更完整的错误码和日志追踪。

