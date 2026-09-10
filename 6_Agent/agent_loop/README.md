# Agent Loop 示例项目

一个基于 for 循环的最简 Agent 实现，展示 LLM + Tool 的完整 ReAct 链路。采用真实 LLM（DeepSeek v4-flash）做决策，工具层全部 Mock。

## 运行

```bash
python main.py
```

## 项目结构

```
agent_loop/
├── main.py                  # 入口：创建 Agent、捕获迭代超限异常
├── .env                     # LLM 配置（API_KEY / MODEL_ID / BASE_URL）
├── agent/
│   ├── state.py             # AgentState dataclass：Agent 的"记忆"容器
│   ├── decision.py          # make_decision()：调 LLM，返回 tool/final 决策
│   ├── agent.py             # CampusAgent.run()：Agent 主循环 + 工具注册表
│   └── exceptions.py        # 异常继承体系：ToolError → Transient(可重试) / Permanent(不可重试) |
└── tools/
    ├── product.py           # search_product()：模拟商品搜索
    ├── calculator.py        # calculator()：数学表达式求值
    └── search.py            # search_knowledge()：模拟知识库查询
```

## 架构链路

```
User 提问
  │
  ▼
CampusAgent.run()  ── for 循环 ◄──────────────────────┐
  │                                                    │
  ▼                                                    │
make_decision(state)  ──→  调 LLM，返回 dict          │
  │                                                    │
  ├─ type="tool" ──► call_tool() ──► 执行工具函数      │
  │                         │                          │
  │                         ▼                          │
  │                      得到 observation              │
  │                         │                          │
  │                  append 进 state.observations      │
  │                         │                          │
  │                     continue ──► 回到 for 循环头    │
  │                                                    │
  └─ type="final" ──► return 最终答案 ──► 结束          │
```

## 核心设计

### 1. AgentState — Agent 的"大脑"

整个 Agent 的状态通过一个 dataclass 贯穿全链路，每轮循环读它、改它：

```python
@dataclass
class AgentState:
    question: str              # 用户原始问题（不变）
    messages: list[dict]       # 对话历史
    current_step: int          # 当前第几轮 for 循环
    observations: list[Any]    # ★ 每一步工具返回的结果都堆在这
    final_answer: Optional[str]
```

LLM 每轮拿到的 prompt 里都会带上完整的历史 observations，所以它知道之前做过什么。

### 2. 工具注册表 — 字符串解耦

LLM 输出的是字符串（比如 `"search_product"`），通过字典映射到真正的函数：

```python
TOOL_MAP = {
    "search_product": search_product,
    "calculator": calculator,
    "search_knowledge": search_knowledge
}
```

这样替换工具实现不会影响 LLM prompt，加新工具只需在 TOOL_MAP 注册 + 在 SYSTEM_PROMPT 描述。

### 3. 异常分层 + Transient / Permanent 区分

**为什么要区分？** 网络抖动重试 3 次可能就好了，但参数是 `None` 重试 100 次也没用。瞎重试 = 垃圾请求放大。

异常继承体系：

```
ToolError (基类)
├── TransientToolError  ← 可重试
│   ├── ToolTimeoutError      网络/执行超时
│   └── ToolNetworkError      网络抖动
└── PermanentToolError  ← 不可重试
    ├── ToolParamError        参数 None / 空 / 类型错
    └── ToolLogicError        eval 除零、语法错、业务逻辑错
```

工具内部的 try/except 按类型分流：

```python
# tools/product.py
def search_product(keyword: str):
    _validate_keyword(keyword)    # 参数校验在 retry 循环外面！
                                  # ToolParamError 直接抛，根本不进 retry

    while retry < MAX_RETRY:
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_do_search, keyword)
                return future.result(timeout=TOOL_TIMEOUT)   # ★ 真正的超时检测

        except FutureTimeout:              # Transient → 重试
            retry += 1
        except TransientToolError as e:    # Transient → 重试
            retry += 1
        except PermanentToolError:         # Permanent → raise，不重试
            raise
        except ZeroDivisionError as e:     # eval 除零 → 包装成 ToolLogicError → 不重试
            raise ToolLogicError(...)
```

两层保护：

| 层级 | 位置 | 作用 |
|---|---|---|
| 工具内部 | `tools/*.py` | 参数校验前置 + 区分异常类型 + 真正超时（ThreadPoolExecutor） |
| Agent 循环层 | `agent/agent.py` | `try/except Exception` 兜住一切，错误存进 observations，LLM 下一轮可见 |

### 4. MAX_ITERATIONS 兜底

`MAX_ITERATIONS = 5`，for 循环耗尽后抛 `AgentMaxIterationsError`，由 main.py 捕获：

```python
for _ in range(MAX_ITERATIONS):
    decision = make_decision(state)
    ...

raise AgentMaxIterationsError("Agent达到最大迭代次数，终止任务")
```

防止模型陷入死循环（比如 Ollama 小模型反复调同一个工具）。

## 一次完整执行示例

```
===== Step 1 =====
LLM Prompt: 用户问题+历史为空 → LLM 决策：调 search_product("二手显示器")
Observation: [{"name":"A显示器","price":399}, {"name":"B显示器","price":450}, {"name":"C显示器","price":699}]

===== Step 2 =====
LLM Prompt: 拿到商品列表 → LLM 决策：筛选<=500的(399+450)，调 calculator("399 + 450")
Observation: 849.0

===== Step 3 =====
LLM Prompt: 有商品列表+有总和 → LLM 算出平均 849/2 = 424.5，输出 final
✅ Final Answer: "500元以内有2台，平均价格424.5元"
```

## 验收特性对照

| 特性 | 实现位置 |
|---|---|
| ① Agent 循环链路 User → Decision → Tool → Observation → Final | `agent/agent.py` + `agent/decision.py` |
| ② for 循环实现 Agent Loop | `agent.py` 第32行 `for _ in range(MAX_ITERATIONS)` |
| ③ 连续调用多个工具 | 状态累积 observations，LLM 看到历史后自动选下一个工具 |
| ④ MAX_ITERATIONS=5 限制循环 | `agent.py` 第8行常量 + 第58行兜底抛异常 |
| ⑤ 工具重试 3 次 + Transient/Permanent 区分 + 真正超时(ThreadPoolExecutor) | `tools/*.py` 内部 retry 循环 + `agent/exceptions.py` 异常继承体系 |
| ⑥ 捕获工具异常不崩程序 | `agent.py` 第44-50行 `try/except Exception`，错误存进 observations |