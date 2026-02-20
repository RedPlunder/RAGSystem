# RAG Agents AIOS Integration Guide

完整教程：如何将Agent集成到AIOS框架中

---

## 📋 目录

1. [概述](#1-概述)
2. [AIOS架构理解](#2-aios架构理解)
3. [Agent开发步骤](#3-agent开发步骤)
4. [Agent集成到AIOS](#4-agent集成到aios)
5. [测试和调试](#5-测试和调试)
6. [发布和分发](#6-发布和分发)
7. [最佳实践](#7-最佳实践)
8. [常见问题](#8-常见问题)

---

## 1. 概述

### 1.1 什么是AIOS？

AIOS（AI Agent Operating System）是一个为LLM代理提供操作系统级支持的框架，包括：

- **AIOS Kernel（内核）**：管理资源（LLM、内存、存储、工具）
- **AIOS SDK（Cerebrum）**：Agent开发和运行的SDK
- **Agent Hub**：Agent市场和分发平台

### 1.2 架构图

```
┌─────────────────────────────────────────────┐
│         Your Agent (应用层)                  │
│  - RAGCoordinator                            │
│  - EmbeddingAgent                            │
│  - RetrievalAgent                            │
│  - GenerationAgent                           │
└────────────────┬────────────────────────────┘
                 │ Syscall Interface
                 ↓
┌─────────────────────────────────────────────┐
│    AIOS Kernel (内核层)                      │
├─────────────────────────────────────────────┤
│  Scheduler  │  LLM Core  │  Memory Manager  │
│  Storage    │  Tool Mgr  │  Context Mgr     │
└─────────────────────────────────────────────┘
                 ↓
         系统资源（GPU、API等）
```

### 1.3 为什么要集成到AIOS？

- ✅ **资源共享**：多个agent共享LLM、内存等资源
- ✅ **调度管理**：AIOS自动管理agent的并发和调度
- ✅ **生态系统**：发布到AIOS Hub供他人使用
- ✅ **可观测性**：完整的性能监控和日志
- ✅ **模块化**：agent可插拔，易于维护和升级

---

## 2. AIOS架构理解

### 2.1 核心组件

#### 2.1.1 AIOS Kernel

位于：`agiresearch/AIOS`

**主要模块**：
- **Scheduler（调度器）**：`aios/scheduler/`
  - FIFO调度器
  - Round Robin调度器
  - 负责处理来自agents的syscall

- **LLM Core（LLM核心）**：`aios/llm_core/`
  - 支持OpenAI、Anthropic、Gemini等
  - 支持本地模型（HuggingFace、vLLM、Ollama）

- **Memory Manager（内存管理）**：`aios/memory/`
  - 短期记忆
  - 长期记忆

- **Storage Manager（存储管理）**：`aios/storage/`
  - 文件存储
  - 向量数据库（Chroma、FAISS）

- **Tool Manager（工具管理）**：`aios/tool/`
  - 外部工具调用
  - MCP Server集成

#### 2.1.2 AIOS SDK (Cerebrum)

位于：`agiresearch/Cerebrum`

**功能**：
- Agent基类定义
- API封装（LLM、Memory、Storage、Tool）
- Agent打包和分发
- AgentManager（下载、加载、运行agents）

### 2.2 Agent与Kernel交互流程

```
1. Agent.run(task)
   ↓
2. Agent调用self.llm_call() / self.storage_query() 等
   ↓
3. 创建Syscall对象（LLMQuery / StorageQuery）
   ↓
4. Syscall添加到全局队列
   ↓
5. Scheduler从队列取出Syscall
   ↓
6. Scheduler调用相应Manager处理
   ↓
7. 结果返回给Syscall
   ↓
8. Agent.run()继续执行
```

### 2.3 Syscall机制

AIOS使用操作系统的syscall概念来实现agent与内核的交互。

**主要Syscall类型**：
- **LLMQuery**：调用LLM生成文本
- **MemoryQuery**：访问记忆系统
- **StorageQuery**：访问存储系统
- **ToolQuery**：调用外部工具

**Syscall执行**：
```python
# 文件：aios/syscall/syscall.py

class SyscallExecutor:
    def execute_request(self, agent_name: str, query: Any):
        # 创建syscall
        syscall = self.create_syscall(agent_name, query)

        # 添加到队列
        if isinstance(query, LLMQuery):
            global_llm_req_queue_add_message(syscall)

        # 等待执行
        syscall.start()
        syscall.join()

        return syscall.get_response()
```

---

## 3. Agent开发步骤

### 3.1 步骤1：设计Agent架构

**问题**：我的RAG系统应该如何拆分为agents？

**方案**：按功能模块拆分
- **EmbeddingAgent**：文档向量化和索引
- **RetrievalAgent**：文档检索和重排序
- **GenerationAgent**：基于上下文的答案生成
- **RAGCoordinator**：协调整个流程

**优势**：
- ✅ 每个agent职责单一
- ✅ 可以独立测试和优化
- ✅ 可以替换特定模块（如更换检索策略）

### 3.2 步骤2：实现Agent基类

创建 `agents/base_agent.py`：

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    def __init__(self, agent_name: str, config: Dict = None):
        self.agent_name = agent_name
        self.config = config or {}

    @abstractmethod
    def run(self, task_input: Any) -> Any:
        """主执行方法"""
        pass

    def llm_call(self, messages: list, model: str = None) -> str:
        """调用LLM（通过AIOS Syscall）"""
        # 在AIOS中，这会创建LLMQuery并通过Syscall执行
        pass

    def storage_query(self, operation: str, **kwargs) -> Any:
        """查询存储（通过AIOS Syscall）"""
        pass

    def memory_query(self, operation: str, **kwargs) -> Any:
        """查询记忆（通过AIOS Syscall）"""
        pass
```

### 3.3 步骤3：实现具体Agent

#### EmbeddingAgent示例

```python
from .base_agent import BaseAgent
import numpy as np
import faiss

class EmbeddingAgent(BaseAgent):
    def __init__(self, agent_name: str = "EmbeddingAgent", config: Dict = None):
        super().__init__(agent_name, config)
        self.index = None

    def run(self, task_input: Dict) -> Dict:
        action = task_input.get("action")

        if action == "create_embeddings":
            return self.create_embeddings(task_input["documents"])
        elif action == "search":
            return self.search(task_input["query"], task_input.get("k", 5))

    def create_embeddings(self, documents: List[str]) -> Dict:
        # 批量生成embeddings
        embeddings = self._batch_embed(documents)

        # 创建FAISS索引
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        return {"status": "success", "num_docs": len(documents)}

    def search(self, query: str, k: int) -> Dict:
        query_emb = self._embed_single(query)
        distances, indices = self.index.search(query_emb, k)
        return {"distances": distances, "indices": indices}
```

#### RAGCoordinator示例

```python
class RAGCoordinator(BaseAgent):
    def __init__(self, agent_name: str = "RAGCoordinator", config: Dict = None):
        super().__init__(agent_name, config)

        # 初始化子agents
        self.embedding_agent = EmbeddingAgent(config=config.get("embedding_config"))
        self.retrieval_agent = RetrievalAgent(config=config.get("retrieval_config"))
        self.generation_agent = GenerationAgent(config=config.get("generation_config"))

    def run(self, task_input: Dict) -> Dict:
        action = task_input.get("action")

        if action == "setup":
            # 设置文档库
            return self.setup(task_input["documents"])

        elif action == "query":
            # 处理查询
            return self.query(
                title=task_input["title"],
                body=task_input["body"]
            )

    def query(self, title: str, body: str) -> Dict:
        # 1. 检索文档
        query_text = f"{title}\n{body}"
        retrieval_results = self.retrieval_agent.run({
            "action": "retrieve",
            "query": query_text,
            "k": 10
        })

        # 2. 生成答案
        context = " ".join(retrieval_results["contents"])
        generation_result = self.generation_agent.run({
            "action": "generate",
            "title": title,
            "body": body,
            "context": context
        })

        return {
            "answer": generation_result["response"],
            "context_ids": retrieval_results["ids"]
        }
```

### 3.4 步骤4：本地测试

创建 `test_agents.py`：

```python
from agents import RAGCoordinator

# 初始化
config = {...}  # 你的配置
coordinator = RAGCoordinator(config=config)

# 设置文档库
coordinator.run({
    "action": "setup",
    "documents": documents,
    "doc_ids": doc_ids
})

# 测试查询
result = coordinator.run({
    "action": "query",
    "title": "Test question",
    "body": "Test body"
})

print(result["answer"])
```

---

## 4. Agent集成到AIOS

### 4.1 方法1：作为Python包（开发阶段）

**适用场景**：本地开发和测试

#### 步骤1：将agents放到AIOS项目中

```bash
AIOS/
├── agents/           # 你的agents包
│   ├── __init__.py
│   ├── base_agent.py
│   ├── embedding_agent.py
│   ├── retrieval_agent.py
│   ├── generation_agent.py
│   └── rag_coordinator.py
├── aios/            # AIOS kernel
├── runtime/
└── ...
```

#### 步骤2：直接导入使用

```python
from agents import RAGCoordinator

agent = RAGCoordinator(config=config)
result = agent.run(task_input)
```

**优点**：
- ✅ 简单直接
- ✅ 适合快速迭代开发

**缺点**：
- ❌ 不符合AIOS的agent分发机制
- ❌ 其他用户无法使用你的agent

### 4.2 方法2：通过AIOS AgentFactory（推荐）

**适用场景**：生产环境和分发

#### 步骤1：创建符合AIOS规范的agent结构

```bash
pyopenagi/agents/yourname/rag_coordinator/
├── agent.py              # Agent主文件
├── config.yaml           # Agent配置
├── requirements.txt      # 依赖列表
└── README.md            # 文档
```

#### 步骤2：实现agent.py

```python
# pyopenagi/agents/yourname/rag_coordinator/agent.py

from agents import RAGCoordinator as BaseRAGCoordinator

class RagCoordinator(BaseRAGCoordinator):
    """
    RAG Coordinator for Kubernetes troubleshooting.

    This agent provides end-to-end RAG functionality including
    document embedding, retrieval, and answer generation.
    """

    def __init__(self, agent_name: str, *args, **kwargs):
        # 从config.yaml加载配置
        config = self._load_config()
        super().__init__(agent_name, config=config)

    def _load_config(self) -> dict:
        """加载agent配置"""
        import yaml
        import os

        config_path = os.path.join(
            os.path.dirname(__file__),
            "config.yaml"
        )

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
```

注意：
- ✅ 类名使用CamelCase（`RagCoordinator`）
- ✅ 文件名使用snake_case（`rag_coordinator`）
- ✅ 必须实现`__init__(self, agent_name: str, *args, **kwargs)`

#### 步骤3：创建config.yaml

```yaml
# pyopenagi/agents/yourname/rag_coordinator/config.yaml

name: "rag_coordinator"
version: "1.0.0"
author: "yourname"
description: "RAG agents for Kubernetes troubleshooting"

# Agent配置
embedding_config:
  embedding_model: "text-embedding-3-small"
  batch_size: 20

retrieval_config:
  rerank_method: "model"

generation_config:
  model: "gpt-4"
  temperature: 0

# 依赖
dependencies:
  - openai>=1.0.0
  - faiss-cpu>=1.7.0
  - transformers>=4.30.0
  - torch>=2.0.0
  - pandas>=1.5.0
  - numpy>=1.24.0
```

#### 步骤4：创建requirements.txt

```txt
openai>=1.0.0
faiss-cpu>=1.7.0
transformers>=4.30.0
torch>=2.0.0
pandas>=1.5.0
numpy>=1.24.0
tenacity>=8.0.0
```

#### 步骤5：创建README.md

```markdown
# RAG Coordinator Agent

Comprehensive RAG system for Kubernetes troubleshooting.

## Features

- Document embedding with OpenAI
- FAISS-based retrieval
- BGE reranker for improved accuracy
- GPT-4 based answer generation

## Usage

\`\`\`python
from pyopenagi.agents.yourname.rag_coordinator.agent import RagCoordinator

agent = RagCoordinator("rag_agent")

# Setup
agent.run({
    "action": "setup",
    "documents": documents,
    "doc_ids": doc_ids
})

# Query
result = agent.run({
    "action": "query",
    "title": "Your question",
    "body": "Detailed question"
})
\`\`\`

## Configuration

See `config.yaml` for configuration options.
```

#### 步骤6：通过AIOS加载和运行

**方式A：使用AgentFactory（Python API）**

```python
from aios.hooks.modules.agent import useFactory

# 初始化factory
submit_agent, await_execution = useFactory(
    log_mode="console",
    max_workers=64
)

# 提交agent
process_id = submit_agent({
    "agent_name": "yourname/rag_coordinator",
    "task_input": {
        "action": "query",
        "title": "Test question",
        "body": "Test body"
    }
})

# 等待结果
result = await_execution(process_id)
print(result)
```

**方式B：使用REST API**

```python
import requests

# 启动AIOS kernel
# bash runtime/launch_kernel.sh

# 提交agent
response = requests.post(
    "http://localhost:8000/agents/submit",
    json={
        "agent_id": "yourname/rag_coordinator",
        "agent_config": {
            "task": {
                "action": "query",
                "title": "Test question",
                "body": "Test body"
            }
        }
    }
)

execution_id = response.json()["execution_id"]

# 查询状态
status = requests.get(
    f"http://localhost:8000/agents/{execution_id}/status"
)

print(status.json())
```

### 4.3 方法3：集成AIOS Syscall（高级）

**适用场景**：需要深度集成AIOS资源管理

#### 修改BaseAgent使用AIOS Syscall

```python
# agents/base_agent.py

from typing import Any, Dict
from aios.llm.api import LLMQuery
from aios.memory.api import MemoryQuery
from aios.storage.api import StorageQuery

class BaseAgent:
    def __init__(self, agent_name: str, config: Dict = None):
        self.agent_name = agent_name
        self.config = config or {}

    def llm_call(self, messages: list, model: str = None) -> str:
        """通过AIOS Syscall调用LLM"""
        from aios.syscall.syscall import SyscallExecutor

        executor = SyscallExecutor()

        # 创建LLMQuery
        query = LLMQuery(
            llms=[{"name": model or "gpt-4", "provider": "openai"}],
            messages=messages,
            action_type="generate"
        )

        # 执行syscall
        response = executor.execute_request(
            agent_name=self.agent_name,
            query=query
        )

        return response["response"]

    def storage_query(self, operation: str, **kwargs) -> Any:
        """通过AIOS Syscall访问存储"""
        from aios.syscall.syscall import SyscallExecutor

        executor = SyscallExecutor()

        # 创建StorageQuery
        query = StorageQuery(
            params=kwargs,
            operation_type=operation
        )

        response = executor.execute_request(
            agent_name=self.agent_name,
            query=query
        )

        return response["response"]
```

**优点**：
- ✅ 完全利用AIOS的资源调度
- ✅ 性能监控和日志
- ✅ 支持并发和优先级

**缺点**：
- ❌ 需要AIOS kernel运行
- ❌ 调试相对复杂

---

## 5. 测试和调试

### 5.1 单元测试

创建 `tests/test_agents.py`：

```python
import pytest
from agents import EmbeddingAgent, RetrievalAgent, GenerationAgent, RAGCoordinator

def test_embedding_agent():
    agent = EmbeddingAgent()

    # 测试embedding创建
    result = agent.run({
        "action": "create_embeddings",
        "documents": ["doc1", "doc2", "doc3"]
    })

    assert result["status"] == "success"
    assert result["num_documents"] == 3

def test_rag_coordinator():
    config = {...}  # 测试配置
    coordinator = RAGCoordinator(config=config)

    # 测试查询
    result = coordinator.run({
        "action": "query",
        "title": "Test",
        "body": "Test body"
    })

    assert "answer" in result
    assert "context_ids" in result

# 运行测试
# pytest tests/test_agents.py -v
```

### 5.2 集成测试

创建 `tests/test_aios_integration.py`：

```python
import pytest
from aios.hooks.modules.agent import useFactory

@pytest.fixture
def agent_factory():
    submit, await_exec = useFactory()
    return submit, await_exec

def test_agent_submission(agent_factory):
    submit_agent, await_execution = agent_factory

    # 提交agent
    process_id = submit_agent({
        "agent_name": "yourname/rag_coordinator",
        "task_input": {"action": "query", ...}
    })

    # 等待结果
    result = await_execution(process_id)

    assert result is not None
    assert "answer" in result
```

### 5.3 调试技巧

#### 1. 启用详细日志

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(name)s] %(asctime)s - %(levelname)s - %(message)s'
)
```

#### 2. 使用断点调试

```python
def run(self, task_input):
    import pdb; pdb.set_trace()  # 设置断点
    # ... 你的代码
```

#### 3. 监控AIOS kernel日志

```bash
# 启动kernel时输出到文件
python -m uvicorn runtime.launch:app --host 0.0.0.0 --port 8000 > aios.log 2>&1

# 监控日志
tail -f aios.log
```

#### 4. 使用AIOS metrics

```python
# 查看agent执行时间
result = executor.execute_request(agent_name, query)

print(f"Waiting time: {result['waiting_times']}")
print(f"Turnaround time: {result['turnaround_times']}")
```

---

## 6. 发布和分发

### 6.1 打包Agent

#### 步骤1：准备文件

确保你的agent目录结构完整：

```bash
pyopenagi/agents/yourname/rag_coordinator/
├── agent.py
├── config.yaml
├── requirements.txt
└── README.md
```

#### 步骤2：创建.agent包

```bash
cd pyopenagi/agents/yourname
tar -czf rag_coordinator_v1.0.0.agent rag_coordinator/

# 验证包内容
tar -tzf rag_coordinator_v1.0.0.agent
```

### 6.2 发布到AIOS Hub

#### 步骤1：注册账号

访问 https://app.aios.foundation/ 注册账号

#### 步骤2：上传Agent

1. 登录AIOS Hub
2. 点击"Upload Agent"
3. 填写信息：
   - Agent名称：`rag_coordinator`
   - 版本：`1.0.0`
   - 描述：简短描述
   - 标签：`rag`, `kubernetes`, `troubleshooting`
4. 上传`.agent`文件
5. 提交审核

#### 步骤3：等待审核

AIOS团队会审核你的agent，通常需要1-3个工作日。

### 6.3 用户安装你的Agent

其他用户可以通过以下方式安装：

```bash
# 方式1：通过AIOS CLI
aios agent install yourname/rag_coordinator

# 方式2：通过Python
from aios.agent_manager import AgentManager

manager = AgentManager()
manager.download_agent("yourname", "rag_coordinator", "1.0.0")
```

---

## 7. 最佳实践

### 7.1 Agent设计原则

1. **单一职责**：每个agent只做一件事
   ```python
   # ✅ 好
   class EmbeddingAgent:
       def run(self, task_input):
           return self.create_embeddings(...)

   # ❌ 不好
   class SuperAgent:
       def run(self, task_input):
           self.create_embeddings(...)
           self.retrieve_documents(...)
           self.generate_answer(...)
   ```

2. **清晰的接口**：使用统一的`run(task_input)`接口
   ```python
   task_input = {
       "action": "query",  # 明确的action字段
       "title": "...",
       "body": "..."
   }
   ```

3. **错误处理**：优雅处理异常
   ```python
   def run(self, task_input):
       try:
           result = self._process(task_input)
           return {"status": "success", "result": result}
       except Exception as e:
           self.logger.error(f"Error: {e}")
           return {"status": "error", "message": str(e)}
   ```

### 7.2 性能优化

1. **批量处理**：
   ```python
   # ✅ 批量embedding
   embeddings = self.batch_embed(documents)

   # ❌ 逐个embedding
   for doc in documents:
       emb = self.embed(doc)
   ```

2. **缓存结果**：
   ```python
   import os

   if os.path.exists(self.cache_path):
       return self.load_cache()
   else:
       result = self.compute()
       self.save_cache(result)
       return result
   ```

3. **异步调用**：
   ```python
   import asyncio

   async def batch_generate(self, queries):
       tasks = [self.generate(q) for q in queries]
       return await asyncio.gather(*tasks)
   ```

### 7.3 可维护性

1. **使用配置文件**：
   ```python
   # ✅ 好
   config = load_yaml("config.yaml")
   agent = RAGCoordinator(config=config)

   # ❌ 不好
   agent = RAGCoordinator(
       model="gpt-4",
       temperature=0,
       batch_size=20,
       ...  # 太多硬编码参数
   )
   ```

2. **版本管理**：
   ```yaml
   # config.yaml
   name: "rag_coordinator"
   version: "1.0.0"  # 使用语义化版本
   ```

3. **文档完善**：
   - 每个agent一个README
   - 清晰的API文档
   - 示例代码

### 7.4 安全性

1. **API密钥管理**：
   ```python
   import os

   # ✅ 从环境变量读取
   api_key = os.getenv("OPENAI_API_KEY")

   # ❌ 硬编码
   api_key = "sk-..."  # 永远不要这样做！
   ```

2. **输入验证**：
   ```python
   def run(self, task_input):
       # 验证必需字段
       if "action" not in task_input:
           raise ValueError("Missing required field: action")

       # 验证action值
       valid_actions = ["query", "setup", "batch_query"]
       if task_input["action"] not in valid_actions:
           raise ValueError(f"Invalid action: {task_input['action']}")
   ```

---

## 8. 常见问题

### Q1: Agent找不到？

**问题**：
```python
ModuleNotFoundError: No module named 'pyopenagi.agents.yourname.rag_coordinator'
```

**解决方案**：
1. 检查目录结构是否正确
2. 确保`agent.py`存在
3. 检查类名是否为CamelCase
4. 尝试重新安装agent：
   ```bash
   aios agent uninstall yourname/rag_coordinator
   aios agent install yourname/rag_coordinator
   ```

### Q2: Syscall执行失败？

**问题**：
```python
RuntimeError: AIOS kernel not running
```

**解决方案**：
1. 确保AIOS kernel已启动：
   ```bash
   bash runtime/launch_kernel.sh
   ```
2. 检查kernel端口是否正确（默认8000）
3. 查看kernel日志：
   ```bash
   tail -f uvicorn.log
   ```

### Q3: 依赖安装失败？

**问题**：
```bash
ERROR: Could not find a version that satisfies the requirement faiss-gpu
```

**解决方案**：
1. 使用CPU版本：`faiss-cpu`
2. 或者根据系统安装GPU版本：
   ```bash
   # For CUDA 11.x
   pip install faiss-gpu

   # For CUDA 12.x
   conda install -c pytorch faiss-gpu
   ```

### Q4: Token限制问题？

**问题**：
```python
openai.error.InvalidRequestError: This model's maximum context length is 8192 tokens
```

**解决方案**：
1. 截断输入：
   ```python
   def truncate_text(text, max_tokens=6000):
       # 简单估算：1 token ≈ 4 chars
       max_chars = max_tokens * 4
       return text[:max_chars]
   ```
2. 使用更大上下文的模型：
   ```yaml
   generation_config:
     model: "gpt-4-turbo"  # 128K context
   ```

### Q5: 如何更新已发布的Agent？

**步骤**：
1. 修改代码
2. 更新`config.yaml`中的版本号：
   ```yaml
   version: "1.1.0"  # 从1.0.0升级到1.1.0
   ```
3. 重新打包：
   ```bash
   tar -czf rag_coordinator_v1.1.0.agent rag_coordinator/
   ```
4. 上传到AIOS Hub并标注更新日志

### Q6: 如何调试Agent性能？

**方法**：
1. 启用时间追踪：
   ```python
   import time

   start = time.time()
   result = self.agent.run(task)
   elapsed = time.time() - start

   print(f"Execution time: {elapsed:.2f}s")
   ```

2. 使用profiler：
   ```python
   import cProfile
   import pstats

   profiler = cProfile.Profile()
   profiler.enable()

   # 你的代码
   result = agent.run(task)

   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)  # 打印前20个最耗时的函数
   ```

3. 监控AIOS metrics：
   ```python
   result = executor.execute_request(agent_name, query)

   print(f"Queue waiting time: {result['waiting_times']}")
   print(f"Execution time: {result['turnaround_times']}")
   ```

---

## 9. 完整示例

### 9.1 从零到部署的完整流程

假设你要创建一个新的agent "my-rag-agent"。

#### 步骤1：创建项目结构

```bash
mkdir -p pyopenagi/agents/yourname/my_rag_agent
cd pyopenagi/agents/yourname/my_rag_agent
```

#### 步骤2：实现agent.py

```python
# agent.py
import sys
import os

# Add path to import base agents
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from agents import RAGCoordinator as BaseRAGCoordinator

class MyRagAgent(BaseRAGCoordinator):
    """My custom RAG agent for specific domain."""

    def __init__(self, agent_name: str, *args, **kwargs):
        config = {
            "embedding_config": {
                "embedding_model": "text-embedding-3-small",
                "batch_size": 20
            },
            "retrieval_config": {
                "rerank_method": "model"
            },
            "generation_config": {
                "model": "gpt-4",
                "temperature": 0
            }
        }
        super().__init__(agent_name, config=config)
```

#### 步骤3：创建配置文件

```yaml
# config.yaml
name: "my_rag_agent"
version: "1.0.0"
author: "yourname"
description: "Custom RAG agent for my use case"
dependencies:
  - openai>=1.0.0
  - faiss-cpu>=1.7.0
```

#### 步骤4：创建依赖文件

```txt
# requirements.txt
openai>=1.0.0
faiss-cpu>=1.7.0
transformers>=4.30.0
torch>=2.0.0
pandas>=1.5.0
numpy>=1.24.0
```

#### 步骤5：测试

```python
# test_local.py
from pyopenagi.agents.yourname.my_rag_agent.agent import MyRagAgent

agent = MyRagAgent("test_agent")

# 测试
result = agent.run({
    "action": "query",
    "title": "Test",
    "body": "Test body"
})

print(result)
```

#### 步骤6：通过AIOS运行

```python
# test_aios.py
from aios.hooks.modules.agent import useFactory

submit_agent, await_execution = useFactory()

# 提交
process_id = submit_agent({
    "agent_name": "yourname/my_rag_agent",
    "task_input": {
        "action": "query",
        "title": "Test",
        "body": "Test"
    }
})

# 等待
result = await_execution(process_id)
print(result)
```

#### 步骤7：打包发布

```bash
cd pyopenagi/agents/yourname
tar -czf my_rag_agent_v1.0.0.agent my_rag_agent/

# 上传到AIOS Hub
```

---

## 10. 总结

### 10.1 关键要点

1. **模块化设计**：将复杂系统拆分为多个专职agent
2. **统一接口**：使用`run(task_input)`作为统一入口
3. **配置驱动**：通过配置文件管理参数
4. **错误处理**：优雅处理异常和边界情况
5. **性能优化**：批量处理、缓存、异步调用
6. **文档完善**：清晰的README和示例代码

### 10.2 学习资源

- **AIOS文档**：https://docs.aios.foundation/
- **AIOS GitHub**：https://github.com/agiresearch/AIOS
- **Cerebrum SDK**：https://github.com/agiresearch/Cerebrum
- **Discord社区**：https://discord.gg/B2HFxEgTJX

### 10.3 下一步

1. ✅ 完成本地agent开发和测试
2. ✅ 集成到AIOS框架
3. ✅ 发布到AIOS Hub
4. 🚀 持续优化和维护
5. 🌟 收集用户反馈并迭代

---

**祝你开发顺利！如有问题，欢迎在AIOS Discord社区提问。** 🎉
