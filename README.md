# everyWriter

each one could writer your own story by ai which can share to public.

## Mature ReAct + Reflexion Agent

这个仓库现在包含一个可直接学习和复用的 **ReAct + Reflexion Agent** 最小成熟实现：

- `everywriter_agent/react_reflexion.py`：核心 Agent、Tool、Memory、Result 数据结构。
- `tests/test_react_reflexion.py`：用脚本化 LLM 验证 ReAct 循环、Reflexion 重试和异常工具处理。
- `examples/openai_adapter_example.py`：展示如何把任意 LLM callable 接入 OpenAI SDK。

### 底层逻辑

ReAct 和 Reflexion 解决的是两个层面的问题：

1. **ReAct = Reason + Act**
   - LLM 先输出 `Thought`，说明当前判断和下一步计划。
   - 如果需要外部信息或计算，输出 `Action` 和 `Action Input`。
   - Agent 执行工具，把结果写回 `Observation`。
   - LLM 基于新的 `Observation` 继续推理，直到输出 `Final Answer`。

2. **Reflexion = 失败后的自我复盘记忆**
   - 如果答案被 `success_checker` 判定失败、工具不存在、重复动作过多，或达到最大轮数仍没有答案，Agent 会调用 Reflexion。
   - Reflexion 不直接回答问题，而是生成一条短的“下次怎么做更好”的经验。
   - 经验写入 `ReflexionMemory`，下一轮 ReAct prompt 会带上这些经验，从而改变策略。

### 快速使用

```python
from everywriter_agent import ReActReflexionAgent, Tool


def fake_llm(prompt: str) -> str:
    # 生产环境中替换为 OpenAI / 本地模型 / 任意推理服务。
    if "Observation:" not in prompt:
        return "Thought: Need exact math.\nAction: calculator\nAction Input: 2 + 3"
    return "Thought: I have the result.\nFinal Answer: 5"


def calculator(expression: str) -> str:
    return str(eval(expression, {"__builtins__": {}}, {}))

agent = ReActReflexionAgent(
    llm=fake_llm,
    tools=[Tool("calculator", "Evaluate basic arithmetic.", calculator)],
)

result = agent.run("What is 2 + 3?")
print(result.answer)      # 5
print(result.steps)       # 可审计完整轨迹
print(result.reflections) # 失败复盘记忆
```

### 设计要点

- **LLM 边界极简**：只要求 `llm(prompt: str) -> str`，方便替换任何模型供应商。
- **工具边界安全**：工具统一返回 `ToolResult`，异常会转为 `Observation: ERROR...`，不会打断主循环。
- **轨迹可审计**：`AgentResult.steps` 保存每一步 Thought、Action、Action Input、Observation。
- **可控终止**：`max_iterations`、`max_reflections`、重复动作检测和 `success_checker` 防止无限循环。
- **可测试**：测试不依赖真实模型，用 `ScriptedLLM` 固定输出，便于学习和回归。

### 运行测试

```bash
python -m unittest discover -s tests
```
