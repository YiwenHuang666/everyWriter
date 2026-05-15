"""A production-ready, dependency-free ReAct + Reflexion agent skeleton.

The module intentionally keeps the LLM boundary as a simple callable so it can be
wired to OpenAI, Anthropic, a local model, or a deterministic fake in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol
import re
import time


class LLM(Protocol):
    """Minimal language-model interface used by the agent."""

    def __call__(self, prompt: str) -> str:
        """Return the model completion for ``prompt``."""


@dataclass(frozen=True)
class ToolResult:
    """Normalized output from a tool invocation."""

    ok: bool
    content: str
    error: str | None = None

    @classmethod
    def success(cls, content: object) -> "ToolResult":
        return cls(ok=True, content=str(content))

    @classmethod
    def failure(cls, error: object) -> "ToolResult":
        return cls(ok=False, content="", error=str(error))

    def as_observation(self) -> str:
        if self.ok:
            return self.content
        return f"ERROR: {self.error}"


@dataclass(frozen=True)
class Tool:
    """A callable capability exposed to the ReAct loop."""

    name: str
    description: str
    func: Callable[[str], object]

    def run(self, tool_input: str) -> ToolResult:
        try:
            return ToolResult.success(self.func(tool_input))
        except Exception as exc:  # pragma: no cover - defensive boundary
            return ToolResult.failure(f"{type(exc).__name__}: {exc}")


@dataclass(frozen=True)
class AgentStep:
    """One Thought/Action/Observation turn in the ReAct trajectory."""

    thought: str
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None


@dataclass(frozen=True)
class AgentResult:
    """Final response and diagnostics for a completed agent run."""

    answer: str
    success: bool
    steps: tuple[AgentStep, ...]
    reflections: tuple[str, ...]
    iterations: int
    elapsed_seconds: float


@dataclass
class ReflexionMemory:
    """Stores compact lessons learned from failed or low-confidence attempts."""

    max_items: int = 8
    _items: list[str] = field(default_factory=list)

    def add(self, reflection: str) -> None:
        cleaned = " ".join(reflection.strip().split())
        if not cleaned:
            return
        self._items.append(cleaned)
        del self._items[:-self.max_items]

    def render(self) -> str:
        if not self._items:
            return "No prior reflections."
        return "\n".join(f"- {item}" for item in self._items)

    def snapshot(self) -> tuple[str, ...]:
        return tuple(self._items)


class ReActReflexionAgent:
    """Runs an inspectable ReAct loop with optional Reflexion self-improvement.

    Expected model protocol per iteration::

        Thought: explain the next reasoning move
        Action: tool_name
        Action Input: argument for the tool

    To stop, the model emits::

        Thought: ...
        Final Answer: answer to the user

    Reflexion is triggered when the loop exceeds ``max_iterations``, calls an
    unknown tool, repeats an action too many times, or a caller-provided
    ``success_checker`` rejects the final answer.
    """

    FINAL_RE = re.compile(r"Final Answer\s*:\s*(?P<answer>.*)", re.I | re.S)
    FIELD_RE = re.compile(
        r"Thought\s*:\s*(?P<thought>.*?)(?:\nAction\s*:\s*(?P<action>.*?)\nAction Input\s*:\s*(?P<input>.*)|\nFinal Answer\s*:)",
        re.I | re.S,
    )

    def __init__(
        self,
        llm: LLM,
        tools: Iterable[Tool],
        *,
        memory: ReflexionMemory | None = None,
        max_iterations: int = 8,
        max_reflections: int = 2,
        success_checker: Callable[[str], bool] | None = None,
    ) -> None:
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.memory = memory or ReflexionMemory()
        self.max_iterations = max_iterations
        self.max_reflections = max_reflections
        self.success_checker = success_checker

    def run(self, task: str) -> AgentResult:
        """Solve ``task`` and return answer plus the full reasoning trajectory."""

        started = time.monotonic()
        all_steps: list[AgentStep] = []
        reflection_count = 0
        last_answer = ""
        failure_reason = ""

        while reflection_count <= self.max_reflections:
            steps: list[AgentStep] = []
            seen_actions: dict[tuple[str, str], int] = {}

            for iteration in range(1, self.max_iterations + 1):
                response = self.llm(self._build_react_prompt(task, steps))
                final = self._parse_final_answer(response)
                if final is not None:
                    last_answer = final.strip()
                    success = bool(last_answer) and (
                        self.success_checker(last_answer)
                        if self.success_checker is not None
                        else True
                    )
                    all_steps.extend(steps)
                    if success:
                        return AgentResult(
                            answer=last_answer,
                            success=True,
                            steps=tuple(all_steps),
                            reflections=self.memory.snapshot(),
                            iterations=len(all_steps),
                            elapsed_seconds=time.monotonic() - started,
                        )
                    failure_reason = "The final answer did not pass the success checker."
                    break

                step = self._parse_action(response)
                if not step.action:
                    failure_reason = "The model did not produce a parseable Action or Final Answer."
                    break

                if step.action not in self.tools:
                    step = AgentStep(
                        thought=step.thought,
                        action=step.action,
                        action_input=step.action_input,
                        observation=f"ERROR: unknown tool '{step.action}'. Available tools: {', '.join(self.tools)}",
                    )
                    steps.append(step)
                    failure_reason = step.observation or "Unknown tool."
                    break

                key = (step.action, step.action_input or "")
                seen_actions[key] = seen_actions.get(key, 0) + 1
                if seen_actions[key] > 2:
                    step = AgentStep(
                        thought=step.thought,
                        action=step.action,
                        action_input=step.action_input,
                        observation="ERROR: repeated the same action too many times; reflect and try a new strategy.",
                    )
                    steps.append(step)
                    failure_reason = step.observation or "Repeated action."
                    break

                result = self.tools[step.action].run(step.action_input or "")
                steps.append(
                    AgentStep(
                        thought=step.thought,
                        action=step.action,
                        action_input=step.action_input,
                        observation=result.as_observation(),
                    )
                )
            else:
                failure_reason = f"Reached max_iterations={self.max_iterations} without a final answer."

            all_steps.extend(steps)
            if reflection_count >= self.max_reflections:
                break
            reflection = self._reflect(task, steps, failure_reason, last_answer)
            self.memory.add(reflection)
            reflection_count += 1

        return AgentResult(
            answer=last_answer or failure_reason,
            success=False,
            steps=tuple(all_steps),
            reflections=self.memory.snapshot(),
            iterations=len(all_steps),
            elapsed_seconds=time.monotonic() - started,
        )

    def _build_react_prompt(self, task: str, steps: list[AgentStep]) -> str:
        tool_text = "\n".join(f"- {name}: {tool.description}" for name, tool in self.tools.items())
        scratchpad = self._render_steps(steps) or "(empty)"
        return f"""You are a careful ReAct agent. Use tools when facts or computation are needed.

Task:
{task}

Available tools:
{tool_text}

Reflexion memory (lessons from previous attempts):
{self.memory.render()}

Scratchpad so far:
{scratchpad}

Respond with exactly one of these formats:
Thought: <what you know and what to do next>
Action: <one tool name>
Action Input: <tool input>

OR
Thought: <why you are done>
Final Answer: <concise answer for the user>
"""

    def _reflect(
        self,
        task: str,
        steps: list[AgentStep],
        failure_reason: str,
        last_answer: str,
    ) -> str:
        prompt = f"""You are the Reflexion module for a ReAct agent.

Task: {task}
Failure reason: {failure_reason}
Last answer: {last_answer or '(none)'}
Trajectory:
{self._render_steps(steps) or '(empty)'}

Write one short, reusable lesson that will help the next attempt succeed. Focus on strategy, missed constraints, and tool usage. Do not solve the task directly.
Reflection:"""
        return self.llm(prompt).strip()

    @classmethod
    def _parse_final_answer(cls, text: str) -> str | None:
        match = cls.FINAL_RE.search(text)
        if match:
            return match.group("answer")
        return None

    @classmethod
    def _parse_action(cls, text: str) -> AgentStep:
        match = cls.FIELD_RE.search(text.strip())
        if not match:
            return AgentStep(thought=text.strip())
        return AgentStep(
            thought=(match.group("thought") or "").strip(),
            action=(match.group("action") or "").strip() or None,
            action_input=(match.group("input") or "").strip() or None,
        )

    @staticmethod
    def _render_steps(steps: list[AgentStep]) -> str:
        rendered: list[str] = []
        for index, step in enumerate(steps, start=1):
            rendered.append(f"Step {index}")
            rendered.append(f"Thought: {step.thought}")
            if step.action:
                rendered.append(f"Action: {step.action}")
                rendered.append(f"Action Input: {step.action_input or ''}")
            if step.observation is not None:
                rendered.append(f"Observation: {step.observation}")
        return "\n".join(rendered)
