# NexusKB LangGraph 状态图

本文档记录当前 Agentic RAG LangGraph 的控制流与关键 `RagState` 字段。主实现位于 `backend/app/rag/agentic_rag_graph.py`。

```mermaid
flowchart TD
  START([START]) --> INIT["initialize\n记录 agentic_graph_initialized"]
  INIT --> UNDERSTAND["understand_request\n读取 history / long-term memory\n产出 AgenticActionDecision"]
  UNDERSTAND --> SAFETY{"safety_check\n规范 action"}

  SAFETY -->|safety_risk| REFUSE["refuse\n生成拒答"]
  SAFETY -->|needs_clarification| CLARIFY["clarify\n请求补充目标 / 范围 / 上下文"]
  SAFETY -->|action = direct_answer| DIRECT["direct_answer\n无需检索直接回答"]
  SAFETY -->|needs_tool / tool_call| TOOL["tool_call\n执行 required_tools"]
  SAFETY -->|needs_retrieval / default| RETRIEVE["retrieve\nplanner\nmetadata_filter_plan\nstrategy_select\nhybrid retrieval"]

  DIRECT --> FINAL["finalize_trace\n记录 agentic_graph_finished"]
  CLARIFY --> FINAL
  REFUSE --> FINAL
  TOOL --> FINAL

  RETRIEVE --> EVAL["evaluate_context\n评估证据充分性 / 质量"]
  EVAL --> DECIDE{"decide_next_action\n根据 evaluator_result 选择下一步"}

  DECIDE -->|rewrite_query / expand_top_k\nretry_count < max_retries| RETRY["apply_retry\n重写查询或扩大 TopK"]
  RETRY --> RETRIEVE

  DECIDE -->|broaden_metadata_filter| BROADEN["broaden_metadata_filter\nhard filter 降级为 soft boost"]
  BROADEN --> RETRIEVE

  DECIDE -->|external_search| WEB["external_search\n当前记录 skipped"]
  WEB --> GENERATE["generate_answer\n生成答案或 insufficient evidence"]

  DECIDE -->|generate| GENERATE
  DECIDE -->|fallback default| GENERATE
  GENERATE --> FINAL
  FINAL --> END([END])

  subgraph STATE["RagState 关键字段"]
    S1["query\noriginal_query / current_query\nrewritten_queries / sub_queries"]
    S2["routing\nintent / action / needs_retrieval\nneeds_tool / safety_risk / source_hints"]
    S3["memory\nhistory / memory_summary\nlong_term_memories / memory_context"]
    S4["retrieval\nplan / strategy\nmetadata_filter_decision\nretrieval_attempts / selected_documents"]
    S5["control\nevaluator_result\nnext_action / retry_count / max_retries"]
    S6["output\nanswer / sources / response_type\nsse_events / error"]
  end

  UNDERSTAND -.reads/writes.-> S2
  UNDERSTAND -.reads.-> S3
  RETRIEVE -.writes.-> S4
  EVAL -.writes.-> S5
  DECIDE -.reads/writes.-> S5
  GENERATE -.writes.-> S6
  FINAL -.reads.-> S6

  classDef startEnd fill:#111827,color:#ffffff,stroke:#111827,stroke-width:2px;
  classDef process fill:#e0f2fe,color:#0f172a,stroke:#0284c7,stroke-width:1.5px;
  classDef decision fill:#fef3c7,color:#0f172a,stroke:#d97706,stroke-width:1.5px;
  classDef terminal fill:#fee2e2,color:#0f172a,stroke:#dc2626,stroke-width:1.5px;
  classDef state fill:#f8fafc,color:#334155,stroke:#94a3b8,stroke-width:1px;

  class START,END startEnd;
  class INIT,UNDERSTAND,RETRIEVE,EVAL,RETRY,BROADEN,WEB,GENERATE,FINAL process;
  class SAFETY,DECIDE decision;
  class DIRECT,CLARIFY,REFUSE,TOOL terminal;
  class S1,S2,S3,S4,S5,S6 state;
```

## 面试讲解口径

这张图体现的是 NexusKB 当前 Agentic RAG 的状态机式编排：请求进入后先初始化 trace，再结合短期对话历史和长期记忆完成意图理解；`safety_check` 负责将请求路由到直接回答、澄清、拒答、工具调用或知识库检索。检索主链路会执行规划、元数据过滤规划、策略选择和混合检索，随后评估上下文证据质量，并根据 `next_action` 决定是否重写查询、扩大 TopK、放宽 metadata filter，或进入最终答案生成。

关键点不是简单串联 RAG chain，而是把请求路由、检索重试、证据评估、答案生成和 trace 记录都放进统一的 `RagState` 中，使 Agent 执行过程可观察、可调试、可扩展。
