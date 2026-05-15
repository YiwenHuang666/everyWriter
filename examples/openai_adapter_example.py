"""Example adapter for using ReActReflexionAgent with the OpenAI Python SDK.

Install the SDK separately if you want to run this file:
    pip install openai

Set OPENAI_API_KEY before running.
"""

from openai import OpenAI

from everywriter_agent import ReActReflexionAgent, Tool


client = OpenAI()


def openai_llm(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )
    return response.output_text


def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/(). %")
    if not set(expression) <= allowed:
        raise ValueError("Only arithmetic expressions are allowed")
    return str(eval(expression, {"__builtins__": {}}, {}))


agent = ReActReflexionAgent(
    llm=openai_llm,
    tools=[Tool("calculator", "Safely evaluate a basic arithmetic expression.", calculator)],
)

if __name__ == "__main__":
    result = agent.run("What is (128 + 64) / 8? Show only the number.")
    print(result.answer)
