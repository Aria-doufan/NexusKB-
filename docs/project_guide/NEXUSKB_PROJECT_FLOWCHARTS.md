# NexusKB 项目流程图总览

本文档汇总当前 NexusKB 项目的整体运行链路与主要模块流程图。图中按线上请求、后台服务、存储组件、离线索引与评估脚本分层描述，便于快速理解系统边界和模块协作。

## 1. 整体系统流程

```mermaid
flowchart LR
    User[用户 / 浏览器] --> Frontend[Vue 3 + Vite 前端]

    Frontend -->|/api/chat /api/sessions /api/debug| FastAPI[FastAPI Backend]
    Frontend -->|/user/*| DjangoUser[Django User Service]
    Frontend -->|/file/*| DjangoFile[Django File Service]

    DjangoUser --> MySQL[(MySQL)]
    DjangoUser --> Redis[(Redis)]
    DjangoFile --> MySQL
    DjangoFile --> Media[(本地媒体/上传文件)]

    FastAPI --> RouterGraph[RouterGraph 路由图]
    FastAPI --> MySQL
    FastAPI --> Redis
    FastAPI --> Chroma[(Chroma 向量库)]

    RouterGraph --> GeneralAgent[LangChain Agent + Tools]
    RouterGraph --> LegacyRAG[通用 RAG]
    RouterGraph --> EnterpriseRAG[Enterprise Agentic RAG]
    RouterGraph --> DirectChat[直接 LLM 对话]

    GeneralAgent --> DeepSeek[DeepSeek / OpenAI 兼容模型]
    DirectChat --> DeepSeek
    LegacyRAG --> Chroma
    EnterpriseRAG --> Chroma
    EnterpriseRAG --> BM25[BM25 本地索引]
    EnterpriseRAG --> Reranker[Qwen3 Reranker 可选]

    FastAPI --> TraceStore[Debug Trace JSONL]
    FastAPI --> MemorySvc[长期记忆服务]
    MemorySvc --> MySQL
    MemorySvc --> Chroma
```

## 2. 在线问答端到端流程

```mermaid
sequenceDiagram
    participant U as 用户
    participant F as Vue 前端
    participant API as FastAPI Chat API
    participant R as RouterGraph
    participant M as 长期记忆服务
    participant ER as EnterpriseRagGraph
    participant LLM as LLM
    participant DB as MySQL / Redis / Chroma

    U->>F: 输入问题
    F->>API: POST /api/chat 或 SSE 流式请求
    API->>API: 解析 JWT、session_id、request_id、debug_id
    API->>R: invoke / stream(state)
    R->>M: 按 user_id + query 召回长期记忆
    M->>DB: MySQL 读取 + Chroma 语义检索
    M-->>R: RagMemoryContext / long_term_memories
    R->>LLM: 路由分类：direct / agent / rag

    alt 企业知识库问题
        R->>ER: 构造 RagState 并运行 Agentic RAG
        ER->>DB: Chroma + BM25 + reranker 检索证据
        ER->>ER: 评估证据、必要时 rewrite / expand_top_k
        ER->>LLM: 带证据和不可信长期记忆生成答案
        ER-->>R: RagResponse + sources + evaluation
    else 工具/通用任务
        R->>LLM: LangChain Agent 调用工具或直接回答
        LLM-->>R: Agent 输出
    end

    R-->>API: RouterResponse / SSE events
    API->>DB: 保存消息、会话、debug trace
    API-->>F: answer / sources / request_id / debug_id
    F-->>U: 渲染 Markdown、引用来源、流式状态
```

## 3. 前端模块流程

```mermaid
flowchart TD
    Entry[front/src/main.js] --> App[App.vue]
    App --> Router[Vue Router]
    App --> Pinia[Pinia 持久化状态]
    App --> Vant[Vant UI]

    Router --> Login[Login.vue]
    Router --> Register[Register.vue]
    Router --> Chat[AIChat.vue]
    Router --> Sessions[Sessions.vue]
    Router --> Profile[Profile.vue]
    Router --> Settings[Settings.vue]
    Router --> My[My.vue]

    Login --> UserStore[store/user.js]
    Register --> UserStore
    Profile --> UserStore
    My --> UserStore

    Chat --> SessionStore[store/session.js]
    Chat --> ChatAPI[/api/chat]
    Sessions --> SessionAPI[/api/sessions]

    UserStore --> UserAPI[/user/*]
    Profile --> FileAPI[/file/*]

    ChatAPI --> Stream[fetch SSE / 流式响应]
    Stream --> Markdown[marked + highlight.js]
    Markdown --> Sanitize[DOMPurify]
    Sanitize --> Render[聊天消息渲染]

    ViteProxy[Vite proxy] -->|/api| FastAPI[FastAPI]
    ViteProxy -->|/user /file| Django[Django UserService]
```

## 4. Django 用户与文件服务流程

```mermaid
flowchart TD
    Request[前端 /user 或 /file 请求] --> DjangoUrls[DjangoUserService/urls.py]

    DjangoUrls --> UserUrls[apps/user/urls.py]
    DjangoUrls --> FileUrls[apps/file/urls.py]

    UserUrls --> AuthViews[登录 / 注册 / 用户信息 / Token]
    UserUrls --> JWTAuth[JWTAuthentication]
    FileUrls --> FileViews[文件 / 头像上传下载]

    AuthViews --> UserModel[Custom User Model]
    JWTAuth --> TokenCheck[解析 JWT + 黑名单检查]
    TokenCheck --> Redis[(Redis JWT blacklist / cache)]

    UserModel --> MySQL[(MySQL)]
    AuthViews --> Cache[Django cache]
    Cache --> Redis

    FileViews --> UserModel
    FileViews --> Media[(MEDIA_ROOT / uploaded files)]
    FileViews --> MySQL

    AuthViews --> Response[统一 JSON 响应]
    FileViews --> Response
```

## 5. FastAPI 后端入口与 API 模块流程

```mermaid
flowchart TD
    Main[backend/main.py] --> FastAPIApp[FastAPI app]
    FastAPIApp --> Middleware[CORS / RateLimitMiddleware / request context]
    FastAPIApp --> Routers[API Routers]

    Routers --> ChatRouter[app/router/chat.py]
    Routers --> Health[health / readiness]
    Routers --> Debug[debug trace endpoints]

    ChatRouter --> Auth[auth_utils JWT 校验]
    Auth --> Redis[(Redis 黑名单检查)]
    ChatRouter --> ChatService[ChatService]

    ChatService --> RouterGraph[RouterGraph]
    ChatService --> SessionManager[database_session_manager]
    ChatService --> ConversationMemory[conversation_memory]
    ChatService --> LongTermMemory[long_term_memory_service]

    SessionManager --> MySQL[(MySQL sessions/messages)]
    ConversationMemory --> MySQL
    LongTermMemory --> MySQL
    LongTermMemory --> Chroma[(Chroma long_term_memories)]

    RouterGraph --> Agent[Agent]
    RouterGraph --> LegacyRAG[Legacy RAG]
    RouterGraph --> EnterpriseRAG[EnterpriseRagGraph]

    ChatService --> Response[RouterResponse / SSE]
```

## 6. RouterGraph 路由状态机

```mermaid
stateDiagram-v2
    [*] --> Initialize
    Initialize --> LoadContext: session_id / user_id
    LoadContext --> RecallMemory: long-term memory enabled
    RecallMemory --> RouteIntent
    LoadContext --> RouteIntent: no memory

    RouteIntent --> ValidateDecision
    ValidateDecision --> DirectChat: route = chat
    ValidateDecision --> AgentTools: route = agent
    ValidateDecision --> GeneralRAG: route = rag / legacy
    ValidateDecision --> EnterpriseRAG: enterprise intent / knowledge query

    DirectChat --> SaveAssistantMessage
    AgentTools --> SaveAssistantMessage
    GeneralRAG --> SaveAssistantMessage
    EnterpriseRAG --> SaveAssistantMessage

    SaveAssistantMessage --> ExtractMemory: eligible conversation
    ExtractMemory --> PersistMemory
    PersistMemory --> [*]
    SaveAssistantMessage --> [*]: memory skipped
```

## 7. LangChain Agent 与工具调用流程

```mermaid
flowchart TD
    RouterGraph --> AgentInvoke[agent.py invoke / stream]
    AgentInvoke --> Prompt[系统提示词 + 对话历史 + 长期记忆]
    Prompt --> LLM[ChatOpenAI / DeepSeek]
    LLM --> Decision{需要工具?}

    Decision -->|否| FinalAnswer[自然语言回答]
    Decision -->|是| ToolCall[工具调用]

    ToolCall --> AgentTools[agent_tools.py]
    AgentTools --> SearchTool[搜索/知识工具]
    AgentTools --> FileTool[文件相关工具]
    AgentTools --> OtherTools[其他业务工具]

    SearchTool --> ToolResult[工具结果]
    FileTool --> ToolResult
    OtherTools --> ToolResult
    ToolResult --> LLM
    LLM --> FinalAnswer

    FinalAnswer --> StreamOrReturn[SSE 增量事件或普通响应]
```

## 8. Enterprise Agentic RAG 流程

```mermaid
flowchart TD
    Start[RagState] --> InitTrace[initialize_trace]
    InitTrace --> Planner[planner: 任务类型 / 证据需求]
    Planner --> Strategy[strategy_select: hybrid / top_k / reranker / decompose]
    Strategy --> NeedDecompose{use_decompose?}

    NeedDecompose -->|否| Retrieve[单查询 hybrid retrieval]
    NeedDecompose -->|是| Decompose[decompose_query]

    Decompose --> DecomposeOK{子查询有效?}
    DecomposeOK -->|否| FallbackSingle[清空 stale sub_queries 并回退单查询]
    DecomposeOK -->|是| RetrieveSub[逐个子查询检索]

    Retrieve --> Evaluate[evaluate_context]
    FallbackSingle --> Evaluate
    RetrieveSub --> MergeScores[merge_decomposed_scores]
    MergeScores --> Evaluate

    Evaluate --> Enough{证据足够且 ACL 允许?}
    Enough -->|是| Generate[generate_answer]
    Enough -->|否| Retry{还有 retry budget?}

    Retry -->|rewrite_query| Rewrite[rewrite_query]
    Retry -->|expand_top_k| Expand[expand_top_k]
    Retry -->|无| Insufficient[build_insufficient_evidence]

    Rewrite --> RetrieveAgain[retrieve]
    Expand --> RetrieveAgain
    RetrieveAgain --> Evaluate

    Generate --> Sources[构造 RagSource]
    Insufficient --> EmptySources[清空 sources]
    Sources --> Finalize[finalize_trace + RagResponse]
    EmptySources --> Finalize
```

## 9. Enterprise RAG 检索服务流程

```mermaid
flowchart TD
    Query[查询 + source_hints + rag_intent] --> Dense[Chroma dense child-chunk search]
    Query --> BM25[BM25 lexical search]

    Dense --> DenseCandidates[向量候选]
    BM25 --> BM25Candidates[词法候选]

    DenseCandidates --> Fusion[RRF 融合]
    BM25Candidates --> Fusion
    Fusion --> SourceBoost[source_hints soft boost]
    SourceBoost --> CandidateTopK[候选截断]

    CandidateTopK --> NeedRerank{intent/confidence 需要 reranker?}
    NeedRerank -->|是| ParentText[加载 parent chunk text]
    ParentText --> Rerank[Qwen3 reranker]
    Rerank --> RankedParents[重排后父块]
    NeedRerank -->|否| RankedParents

    RankedParents --> ExpandParent[父块扩展 / sibling chunks]
    ExpandParent --> FormatDocs[格式化 parent_doc_id / parent_chunk_id / title / score]
    FormatDocs --> ReturnDocs[返回 EnterpriseRAG 文档]
```

## 10. 通用 RAG 与文档上传流程

```mermaid
flowchart TD
    Upload[用户上传文档] --> FileAPI[FastAPI / 文档上传接口]
    FileAPI --> Validate[文件类型/大小校验]
    Validate --> Parse[文档解析]
    Parse --> Split[文本切分]
    Split --> Embed[Embedding]
    Embed --> ChromaUser[(Chroma 用户文档集合)]
    FileAPI --> MySQL[(MySQL 文件/会话元数据)]

    Ask[用户问题] --> RouterGraph
    RouterGraph --> LegacyRAG[rag_service]
    LegacyRAG --> VectorStore[vector_store]
    VectorStore --> ChromaUser
    ChromaUser --> RetrievedDocs[相关文档片段]
    RetrievedDocs --> LLM[LLM 生成答案]
    LLM --> Answer[回答 + 引用]
```

## 11. 长期记忆与会话记忆流程

```mermaid
flowchart TD
    ChatTurn[用户/助手消息] --> ConversationMemory[conversation_memory]
    ConversationMemory --> MySQLMessages[(MySQL chat_messages)]

    ChatTurn --> Extractor{是否抽取长期记忆?}
    Extractor -->|否| End[结束]
    Extractor -->|是| LLMExtract[LLM 抽取事实/偏好/项目上下文]

    LLMExtract --> Normalize[规范化 memory_type / hash / metadata]
    Normalize --> DuplicateCheck[hash + Chroma 语义重复检查]
    DuplicateCheck -->|重复| Skip[跳过或更新]
    DuplicateCheck -->|新记忆| PersistSQL[写入 MySQL long_term_memories]
    PersistSQL --> EmbedMemory[Embedding]
    EmbedMemory --> ChromaMemory[(Chroma long_term_memories)]

    Query[新问题] --> Recall[long_term_memory_service.search]
    Recall --> ChromaMemory
    Recall --> MySQLActive[MySQL active memory 校验]
    MySQLActive --> Sanitize[角色前缀净化 + 不可信上下文声明]
    Sanitize --> PromptContext[注入 Agent / EnterpriseRAG memory_context]
```

## 12. 数据存储关系流程

```mermaid
flowchart LR
    subgraph MySQL[MySQL]
        Users[users]
        Sessions[chat_sessions]
        Messages[chat_messages]
        LongMem[long_term_memories]
        FileMeta[file metadata]
    end

    subgraph Redis[Redis]
        JWTBlacklist[JWT blacklist]
        UserCache[user/cache]
        RateLimit[rate limit counters]
        Readiness[readiness dependency]
    end

    subgraph Chroma[Chroma]
        UserDocs[user document vectors]
        EnterpriseChunks[enterprise benchmark chunks]
        MemoryVectors[long_term_memory vectors]
    end

    subgraph Files[Local Files]
        Media[uploaded media]
        TraceJSONL[rag_debug_traces/*.jsonl]
        BM25Index[BM25 / evaluation artifacts]
    end

    Django[Django UserService] --> Users
    Django --> FileMeta
    Django --> JWTBlacklist
    Django --> UserCache
    Django --> Media

    FastAPI[FastAPI Backend] --> Sessions
    FastAPI --> Messages
    FastAPI --> LongMem
    FastAPI --> RateLimit
    FastAPI --> UserDocs
    FastAPI --> EnterpriseChunks
    FastAPI --> MemoryVectors
    FastAPI --> TraceJSONL
    EnterpriseRAG[EnterpriseRAG] --> BM25Index
```

## 13. 离线索引与评估流程

```mermaid
flowchart TD
    SourceDocs[企业文档 / benchmark 原始数据] --> Prepare[prepare_enterprise_rag_bench.py]
    Prepare --> Golden[golden questions / answer facts / evidence groups]
    Prepare --> Chunks[enterprise chunks JSONL]

    Chunks --> IndexChroma[index_enterprise_chunks_chroma.py]
    IndexChroma --> EnterpriseCollection[(Chroma enterprise chunks)]
    Chunks --> BuildBM25[BM25 索引构建]
    BuildBM25 --> BM25Artifacts[(BM25 artifacts)]

    Golden --> EvalHybrid[evaluate_enterprise_hybrid_retrieval.py]
    EnterpriseCollection --> EvalHybrid
    BM25Artifacts --> EvalHybrid
    EvalHybrid --> Metrics[hit@k / precision / recall / f1 / rr / evidence_coverage@k]
    EvalHybrid --> Reports[CSV / JSON summary]

    MemoryCases[memory_eval_golden_cases.jsonl] --> EvalMemory[evaluate_long_term_memory.py]
    EvalMemory --> MemoryMetrics[长期记忆召回/去重/相关性评估]
```

## 14. 启动与开发环境流程

```mermaid
flowchart TD
    Dev[开发者] --> StartScript[start-dev.ps1]
    StartScript --> Env[conda nexuskb / Node / Django settings]

    Env --> StartFastAPI[启动 FastAPI backend]
    Env --> StartDjango[启动 DjangoUserService]
    Env --> StartFrontend[启动 Vite frontend]
    Env --> StartRedis[确认 Redis 可用]
    Env --> StartMySQL[确认 MySQL 可用]
    Env --> StartChroma[确认 Chroma 数据目录/集合]

    StartFastAPI --> BackendHealth[/health]
    StartDjango --> UserHealth[/user endpoints]
    StartFrontend --> Browser[浏览器访问前端]

    Browser --> ViteProxy[Vite proxy]
    ViteProxy --> BackendHealth
    ViteProxy --> UserHealth

    BackendHealth --> Ready{依赖可用?}
    UserHealth --> Ready
    Ready -->|是| DevReady[本地开发可用]
    Ready -->|否| FixDeps[检查 Redis / MySQL / API keys / Chroma]
```

## 15. 安全与失败处理流程

```mermaid
flowchart TD
    Request[进入请求] --> Auth{需要认证?}
    Auth -->|否| RateLimit[限流检查]
    Auth -->|是| JWT[解析 JWT]
    JWT --> Blacklist[Redis blacklist 检查]

    Blacklist --> RedisOK{Redis 可用?}
    RedisOK -->|否| HTTP503[返回 503 Redis unavailable]
    RedisOK -->|是| TokenValid{Token 有效且未拉黑?}
    TokenValid -->|否| HTTP401[返回 401]
    TokenValid -->|是| RateLimit

    RateLimit --> RateRedisOK{Redis 可用?}
    RateRedisOK -->|否| HTTP503
    RateRedisOK -->|是| Limited{超出限流?}
    Limited -->|是| HTTP429[返回 429]
    Limited -->|否| Business[进入业务处理]

    Business --> RAG[Agentic RAG / Agent]
    RAG --> ACL{ACL 或安全标记阻断?}
    ACL -->|是| Insufficient[不生成答案，不返回来源]
    ACL -->|否| Answer[生成答案]

    Business --> MemoryContext[长期记忆上下文]
    MemoryContext --> Sanitized[净化 role prefix + 标记为不可信事实]
    Sanitized --> Prompt[进入 LLM prompt]
```

## 16. 模块索引

| 模块 | 主要职责 | 关键入口 |
| --- | --- | --- |
| Frontend | 页面路由、聊天 UI、SSE 渲染、用户状态 | `front/src/main.js`, `front/src/views/AIChat.vue` |
| Django UserService | 注册登录、JWT、用户资料、文件/头像 | `DjangoUserService/DjangoUserService/urls.py` |
| FastAPI Backend | 聊天 API、会话、RAG/Agent 编排、debug trace | `backend/main.py`, `backend/app/router/chat.py` |
| RouterGraph | 意图路由、上下文加载、RAG/Agent 分流 | `backend/app/agent/router_graph.py` |
| Agent | 通用 LLM 对话与工具调用 | `backend/app/agent/agent.py` |
| Enterprise RAG | 企业知识库 agentic 检索、评估、重试、引用 | `backend/app/rag/enterprise_rag_graph.py` |
| RAG Service | 通用文档向量检索和回答 | `backend/app/rag/rag_service.py` |
| Long-term Memory | 用户长期记忆抽取、存储、召回、删除 | `backend/app/services/long_term_memory.py` |
| Conversation Memory | 会话消息与上下文管理 | `backend/app/services/conversation_memory.py` |
| Evaluation Scripts | 企业 RAG、分解检索、长期记忆评估 | `backend/scripts/*.py` |
| Ops / Startup | 本地启动、部署、依赖健康检查 | `start-dev.ps1`, `docs/ops/deployment.md` |
