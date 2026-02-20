# Multi-Agent System Architecture

## 概述

这是一个基于**Planning（规划）**的多智能体协作系统，而非传统的RAG系统。核心思想是：

```
User Query → Planning Agent → Task Agents → Results
```

## 架构设计

### 核心理念

```
┌─────────────────────────────────────────────────────────┐
│                      User Query                         │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│               Planning LLM Agent                        │
│  - Analyzes user intent                                 │
│  - Creates execution plan                               │
│  - Decides which agents to call                         │
└────────────────────┬────────────────────────────────────┘
                     ↓
        ┌────────────┴────────────┐
        ↓                         ↓
┌──────────────┐          ┌──────────────┐
│ Task Agent 1 │          │ Task Agent 2 │  ...
│ (Embedding)  │          │ (Retrieval)  │
└──────┬───────┘          └──────┬───────┘
       │                         │
       └────────────┬────────────┘
                    ↓
        ┌───────────────────────┐
        │  Planning Agent       │
        │  Aggregates Results   │
        └───────────┬───────────┘
                    ↓
        ┌───────────────────────┐
        │    Final Result       │
        └───────────────────────┘
```

## 核心组件

### 1. PlanningAgent（规划代理）

**职责**：
- 分析用户查询，理解意图
- 创建执行计划（包含步骤、所需agents、参数）
- 调用task agents执行计划
- 聚合多个agents的结果
- 追踪执行指标

**工作流**：

```python
planning_agent = PlanningAgent()

# 注册可用的task agents
planning_agent.register_agent("EmbeddingAgent", embedding_agent)
planning_agent.register_agent("RetrievalAgent", retrieval_agent)
planning_agent.register_agent("GenerationAgent", generation_agent)

# 分析查询并执行
result = planning_agent.run({
    "execution_mode": "plan_and_execute",
    "query": "User's question here"
})
```

**执行模式**：

1. **plan_only**：仅创建计划，不执行
   ```python
   plan = planning_agent.run({
       "execution_mode": "plan_only",
       "query": "..."
   })
   # 返回：{"steps": [...], "reasoning": "..."}
   ```

2. **plan_and_execute**：创建并执行计划
   ```python
   result = planning_agent.run({
       "execution_mode": "plan_and_execute",
       "query": "..."
   })
   # 返回：{"plan": {...}, "execution": {...}}
   ```

3. **execute_plan**：执行预定义计划
   ```python
   result = planning_agent.run({
       "execution_mode": "execute_plan",
       "plan": custom_plan
   })
   ```

### 2. Task Agents（任务代理）

这些是执行具体任务的agents，由Planning Agent调用：

#### EmbeddingAgent
- **功能**：文档向量化和FAISS索引
- **使用场景**：当需要语义搜索时

#### RetrievalAgent
- **功能**：文档检索和重排序
- **使用场景**：当需要从知识库获取相关信息时

#### GenerationAgent
- **功能**：基于LLM的文本生成
- **使用场景**：需要生成回答、总结、翻译等

## 执行计划格式

Planning Agent创建的计划是JSON格式：

```json
{
  "steps": [
    {
      "step": 1,
      "description": "Search for relevant documentation",
      "agent": "RetrievalAgent",
      "input": {
        "action": "retrieve",
        "queries": ["kubernetes ingress"],
        "k": 10
      },
      "output_key": "retrieved_docs",
      "critical": true
    },
    {
      "step": 2,
      "description": "Generate answer based on context",
      "agent": "GenerationAgent",
      "input": {
        "action": "generate",
        "title": "User question",
        "context": "$context.retrieved_docs"
      },
      "output_key": "final_answer",
      "critical": true
    }
  ],
  "reasoning": "First retrieve relevant docs, then generate answer"
}
```

### 计划字段说明

- **step**: 步骤编号
- **description**: 步骤描述
- **agent**: 要调用的agent名称
- **input**: 传递给agent的参数
- **output_key**: 存储结果的键（用于后续步骤引用）
- **critical**: 如果失败是否中止整个计划

### 上下文引用

后续步骤可以引用前面步骤的结果：

```json
{
  "context": "$context.retrieved_docs"
}
```

这会被替换为前面步骤存储在`retrieved_docs`键中的实际结果。

## 工作流示例

### 示例1：简单查询

```
用户: "How do I configure Kubernetes Ingress?"

Planning Agent 分析:
1. 用户需要Kubernetes配置信息
2. 需要检索相关文档
3. 需要生成结构化答案

执行计划:
Step 1: RetrievalAgent.retrieve(query="kubernetes ingress")
Step 2: GenerationAgent.generate(context=step1_result)

结果: 完整的Ingress配置指南
```

### 示例2：复杂任务

```
用户: "Compare different Kubernetes service types and recommend one for my web app"

Planning Agent 分析:
1. 需要获取多种Service类型的文档
2. 需要理解用户的应用场景
3. 需要比较分析
4. 需要提供推荐

执行计划:
Step 1: RetrievalAgent.retrieve(query="kubernetes service types")
Step 2: RetrievalAgent.retrieve(query="kubernetes web application deployment")
Step 3: GenerationAgent.generate(
    context=step1_result + step2_result,
    task="compare and recommend"
)

结果: 详细比较和推荐
```

## 与传统RAG的区别

### 传统RAG：
```
Query → Retrieve → Generate → Answer
```
- 固定流程
- 每次都检索
- 单一路径

### Planning-based System：
```
Query → Plan → [Dynamic Agent Calls] → Answer
```
- 动态流程
- 根据需求决定是否检索
- 可以调用多个agents
- 支持复杂工作流

## 扩展性

### 添加新的Task Agent

```python
# 1. 创建新agent
class CustomAgent(BaseAgent):
    def run(self, task_input):
        # 实现你的逻辑
        return result

# 2. 注册到Planning Agent
custom_agent = CustomAgent()
planning_agent.register_agent("CustomAgent", custom_agent)

# 3. Planning Agent会自动发现并可以在计划中使用
```

### 自定义Planning策略

```python
class CustomPlanningAgent(PlanningAgent):
    def _construct_planning_prompt(self, query, context, available_agents):
        # 自定义planning prompt
        return custom_prompt

    def _parse_plan(self, plan_text):
        # 自定义plan解析逻辑
        return parsed_plan
```

## 性能监控

Planning Agent 追踪以下指标：

```python
metrics = planning_agent.get_metrics()

print(metrics)
# {
#   "planning_time": 2.5,      # Planning耗时
#   "execution_time": 10.3,    # 执行耗时
#   "total_time": 12.8,        # 总耗时
#   "agents_called": [          # 调用的agents
#       "RetrievalAgent",
#       "GenerationAgent"
#   ]
# }
```

## 最佳实践

### 1. 明确Agent职责
- 每个Task Agent应该只做一件事
- Planning Agent负责协调，不做具体任务

### 2. 设计清晰的接口
```python
# ✅ 好的设计
def run(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
    action = task_input.get("action")
    if action == "specific_task":
        return self.do_task(task_input)

# ❌ 不好的设计
def run(self, **kwargs):
    # 参数不明确
    pass
```

### 3. 使用output_key传递数据
```python
# 在计划中使用output_key
{
  "step": 1,
  "output_key": "search_results",
  ...
}

# 后续步骤引用
{
  "step": 2,
  "input": {
    "data": "$context.search_results"
  }
}
```

### 4. 标记关键步骤
```python
{
  "step": 1,
  "critical": True,  # 失败则中止
  ...
}
```

## 未来扩展方向

1. **并行执行**：支持多个独立步骤并行执行
2. **条件分支**：根据中间结果选择不同路径
3. **循环执行**：支持迭代优化
4. **错误恢复**：自动重试和备选方案
5. **学习优化**：基于历史执行优化planning策略

## 总结

这个架构的核心优势：

- ✅ **灵活性**：动态决定执行流程
- ✅ **可扩展**：轻松添加新agents
- ✅ **可维护**：职责清晰，模块解耦
- ✅ **可观测**：完整的执行追踪
- ✅ **智能化**：LLM驱动的planning

这不是一个固定的RAG系统，而是一个**智能的multi-agent orchestration platform**。
