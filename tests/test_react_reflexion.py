import unittest

from everywriter_agent import ReActReflexionAgent, Tool


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("No scripted LLM responses left")
        return self.responses.pop(0)


class ReActReflexionAgentTest(unittest.TestCase):
    def test_react_tool_then_final_answer(self):
        llm = ScriptedLLM(
            [
                "Thought: Need arithmetic.\nAction: calculator\nAction Input: 2 + 3",
                "Thought: The observation is enough.\nFinal Answer: 5",
            ]
        )
        agent = ReActReflexionAgent(
            llm=llm,
            tools=[Tool("calculator", "Add numbers", lambda text: eval(text, {"__builtins__": {}}, {}))],
        )

        result = agent.run("What is 2 + 3?")

        self.assertTrue(result.success)
        self.assertEqual(result.answer, "5")
        self.assertEqual(result.steps[0].observation, "5")
        self.assertIn("Observation: 5", llm.prompts[1])

    def test_reflexion_after_failed_answer(self):
        llm = ScriptedLLM(
            [
                "Thought: Guessing.\nFinal Answer: wrong",
                "Check the tool result instead of guessing.",
                "Thought: Need exact lookup.\nAction: lookup\nAction Input: answer",
                "Thought: Done.\nFinal Answer: correct",
            ]
        )
        agent = ReActReflexionAgent(
            llm=llm,
            tools=[Tool("lookup", "Return the answer", lambda _: "correct")],
            success_checker=lambda answer: answer == "correct",
            max_reflections=1,
        )

        result = agent.run("Return the expected answer.")

        self.assertTrue(result.success)
        self.assertEqual(result.answer, "correct")
        self.assertEqual(result.reflections, ("Check the tool result instead of guessing.",))
        self.assertIn("Reflexion memory", llm.prompts[2])
        self.assertIn("Check the tool result instead of guessing.", llm.prompts[2])

    def test_unknown_tool_returns_failed_result_with_reflection(self):
        llm = ScriptedLLM(
            [
                "Thought: I need a missing tool.\nAction: search\nAction Input: x",
                "Use only tools listed in the prompt.",
            ]
        )
        agent = ReActReflexionAgent(llm=llm, tools=[], max_reflections=0)

        result = agent.run("Find x")

        self.assertFalse(result.success)
        self.assertIn("unknown tool", result.steps[0].observation)


if __name__ == "__main__":
    unittest.main()
