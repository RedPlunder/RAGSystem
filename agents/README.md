# Planning-based Multi-Agent System for AIOS

**这是一个基于Planning（规划）的智能多代理协作系统，而非传统的RAG系统。**

## 🎯 核心理念

```
User Query → Planning Agent → Dynamic Task Execution → Results
```

Planning Agent分析用户查询，创建执行计划，动态调用所需的Task Agents，最终完成任务。

## 🏗️ 架构

```
┌─────────────────────────────────────┐
│          User Query                 │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│      Planning LLM Agent             │
│  - Analyze intent                   │
│  - Create execution plan            │
│  - Decide which agents to call      │
└──────────────┬──────────────────────┘
               ↓
    ┌──────────┴──────────┐
    ↓                     ↓
┌──────────┐        ┌──────────┐
│ Task     │        │ Task     │  ...
│ Agent 1  │        │ Agent 2  │
└────┬─────┘        └────┬─────┘
     │                   │
     └─────────┬─────────┘
               ↓
    ┌──────────────────┐
    │  Aggregate &     │
    │  Return Results  │
    └──────────────────┘
```

## 📦 组件

### 1. PlanningAgent（核心）

**职责**：智能规划和协调

```python
from agents import PlanningAgent

planning_agent = PlanningAgent(config={
    "planning_model": "gpt-4",
    "planning_temperature": 0
})

# 注册可用的task agents
planning_agent.register_agent("RetrievalAgent", retrieval_agent)
planning_agent.register_agent("GenerationAgent", generation_agent)

# 执行任务
result = planning_agent.run({
    "execution_mode": "plan_and_execute",
    "query": "Your question here"
})
```

**执行模式**：
- `plan_only`: 仅创建计划
- `plan_and_execute`: 创建并执行计划（推荐）
- `execute_plan`: 执行预定义计划

### 2. Task Agents

这些agents执行具体任务，由Planning Agent调用：

#### EmbeddingAgent
- 文档向量化
- FAISS索引管理

#### RetrievalAgent
- 文档检索
- 重排序（tag-based / model-based）

#### GenerationAgent
- LLM文本生成
- 多轮对话
- Token追踪

## 🚀 快速开始

### 基础示例

```python
from agents import PlanningAgent, RetrievalAgent, GenerationAgent

# 1. 初始化agents
planning_agent = PlanningAgent()
retrieval_agent = RetrievalAgent()
generation_agent = GenerationAgent()

# 2. 注册task agents
planning_agent.register_agent("RetrievalAgent", retrieval_agent)
planning_agent.register_agent("GenerationAgent", generation_agent)

# 3. 执行查询
result = planning_agent.run({
    "execution_mode": "plan_and_execute",
    "query": "How to configure Kubernetes Ingress?",
    "context": "User needs to expose a web service"
})

# 4. 查看结果
print(result["execution"]["results"])
```

### 完整示例

参见 [examples/example_planning_workflow.py](examples/example_planning_workflow.py)

```bash
python agents/examples/example_planning_workflow.py
```

## 📋 执行计划格式

Planning Agent自动生成的计划是JSON格式：

```json
{
  "steps": [
    {
      "step": 1,
      "description": "Search for relevant documentation",
      "agent": "RetrievalAgent",
      "input": {
        "action": "retrieve",
        "queries": ["kubernetes ingress"]
      },
      "output_key": "docs",
      "critical": true
    },
    {
      "step": 2,
      "description": "Generate answer",
      "agent": "GenerationAgent",
      "input": {
        "action": "generate",
        "context": "$context.docs"
      },
      "output_key": "answer",
      "critical": true
    }
  ],
  "reasoning": "First retrieve docs, then generate answer"
}
```

### 上下文引用

使用 `$context.<key>` 引用前面步骤的结果：

```json
{
  "input": {
    "context": "$context.retrieved_docs"
  }
}
```

## 🔧 高级用法

### 自定义执行计划

```python
custom_plan = {
    "steps": [
        {
            "step": 1,
            "agent": "RetrievalAgent",
            "input": {"action": "retrieve", ...},
            "output_key": "docs",
            "critical": True
        },
        {
            "step": 2,
            "agent": "GenerationAgent",
            "input": {
                "action": "generate",
                "context": "$context.docs"
            }
        }
    ]
}

result = planning_agent.run({
    "execution_mode": "execute_plan",
    "plan": custom_plan
})
```

### 添加新的Task Agent

```python
from agents import BaseAgent

class CustomAgent(BaseAgent):
    def run(self, task_input):
        # 实现你的逻辑
        action = task_input.get("action")
        if action == "custom_task":
            return self.do_custom_task(task_input)
        return {"result": "done"}

# 注册
custom_agent = CustomAgent()
planning_agent.register_agent("CustomAgent", custom_agent)

# Planning Agent会自动发现并可以在计划中使用
```

### 性能监控

```python
# 获取执行指标
metrics = planning_agent.get_metrics()

print(f"Planning time: {metrics['planning_time']:.2f}s")
print(f"Execution time: {metrics['execution_time']:.2f}s")
print(f"Agents called: {metrics['agents_called']}")

# 重置指标
planning_agent.reset_metrics()
```

## 📚 配置

### config.yaml示例

```yaml
# Planning Agent配置
planning_config:
  planning_model: "gpt-4"
  planning_temperature: 0

# Task Agent配置
embedding_config:
  embedding_model: "text-embedding-3-small"
  batch_size: 20

retrieval_config:
  rerank_method: "model"

generation_config:
  model: "gpt-4"
  temperature: 0
```

## 🆚 与传统RAG的区别

| 特性 | 传统RAG | Planning-based System |
|------|---------|----------------------|
| 流程 | 固定（检索→生成） | 动态（根据需求） |
| 灵活性 | 低 | 高 |
| 复杂任务 | 困难 | 容易 |
| 多agent协作 | 不支持 | 支持 |
| 可扩展性 | 低 | 高 |

**传统RAG**:
```
Query → Retrieve → Generate → Answer
```

**Planning-based System**:
```
Query → Analyze → Plan → [Dynamic Agents] → Aggregate → Answer
```

## 📁 文件结构

```
agents/
├── __init__.py              # 包初始化
├── base_agent.py            # Agent基类
├── planning_agent.py        # 规划协调Agent
├── embedding_agent.py       # 向量化Agent
├── retrieval_agent.py       # 检索Agent
├── generation_agent.py      # 生成Agent
├── config.yaml              # 配置文件
├── ARCHITECTURE.md          # 架构详解
├── README.md                # 本文件
└── examples/                # 示例
    ├── example_planning_workflow.py
    ├── example_single_query.py
    ├── example_batch_query.py
    └── example_aios_integration.py
```

## 💡 使用场景

### 1. 简单查询
```
Query: "How to configure Kubernetes Ingress?"

Plan:
1. RetrievalAgent.retrieve("kubernetes ingress")
2. GenerationAgent.generate(context=step1_result)
```

### 2. 复杂分析
```
Query: "Compare Kubernetes service types and recommend one"

Plan:
1. RetrievalAgent.retrieve("kubernetes service types")
2. RetrievalAgent.retrieve("kubernetes networking")
3. GenerationAgent.generate(
    context=step1_result + step2_result,
    task="compare and recommend"
)
```

### 3. 多步骤任务
```
Query: "Create a deployment guide for my web app"

Plan:
1. RetrievalAgent.retrieve("kubernetes deployment")
2. GenerationAgent.generate(outline)
3. RetrievalAgent.retrieve(based on outline)
4. GenerationAgent.generate(final guide)
```

## 🔌 AIOS集成

### 本地模式

```python
from agents import PlanningAgent

agent = PlanningAgent()
result = agent.run(task_input)
```

### AIOS Kernel模式

参见 [AIOS_INTEGRATION_GUIDE.md](AIOS_INTEGRATION_GUIDE.md)

```python
from aios.hooks.modules.agent import useFactory

submit_agent, await_execution = useFactory()

process_id = submit_agent({
    "agent_name": "yourname/planning_agent",
    "task_input": {"query": "..."}
})

result = await_execution(process_id)
```

## 🛠️ 依赖

```bash
pip install openai faiss-cpu transformers torch pandas numpy tenacity
```

## 📖 文档

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: 详细架构说明
- **[AIOS_INTEGRATION_GUIDE.md](AIOS_INTEGRATION_GUIDE.md)**: AIOS集成完整教程
- **[examples/](examples/)**: 各种示例代码

## 🎓 最佳实践

1. **明确Agent职责**: 每个Task Agent只做一件事
2. **使用output_key**: 在步骤间传递数据
3. **标记关键步骤**: 使用`critical`标记必须成功的步骤
4. **监控性能**: 使用`get_metrics()`追踪执行情况
5. **自定义Planning**: 可以继承`PlanningAgent`自定义规划逻辑

## 🚧 未来扩展

- [ ] 并行执行独立步骤
- [ ] 条件分支（if-else逻辑）
- [ ] 循环执行（迭代优化）
- [ ] 自动错误恢复
- [ ] 基于历史的Planning优化

## 📄 License

Same as AIOS project.

---

**注意**: 虽然保留了`RAGCoordinator`作为向后兼容的别名，但推荐使用`PlanningAgent`这个更准确的名称。

```python
# 旧代码（仍然可用）
from agents import RAGCoordinator
coordinator = RAGCoordinator()

# 新代码（推荐）
from agents import PlanningAgent
planning_agent = PlanningAgent()
```
