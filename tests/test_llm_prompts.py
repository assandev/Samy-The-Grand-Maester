import unittest

from src.llm.prompts import build_pdf_qa_prompt, get_pdf_qa_system_prompt


class PromptTests(unittest.TestCase):
    def test_system_prompt_mentions_pdf_grounding_and_fallback(self) -> None:
        prompt = get_pdf_qa_system_prompt()
        self.assertIn("uploaded PDF", prompt)
        self.assertIn("general knowledge", prompt)
        self.assertIn("Do not invent", prompt)

    def test_user_prompt_includes_context_question_and_fallback_instruction(self) -> None:
        prompt = build_pdf_qa_prompt(
            question="What is this PDF about?",
            context="This PDF explains introductory AI concepts.",
        )
        self.assertIn("PDF context:", prompt)
        self.assertIn("User question:", prompt)
        self.assertIn("general knowledge", prompt)
        self.assertIn("uploaded PDF", prompt)


if __name__ == "__main__":
    unittest.main()
