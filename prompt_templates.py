"""
Prompt Engineering Module
-------------------------
Centralized prompt templates and LLM-calling logic.

This module demonstrates systematic prompt engineering:
- Each task has a role-specified system prompt
- User prompts use variable-substitution with safe defaults
- Outputs are constrained to structured formats (JSON) where downstream
  parsing is needed, reducing hallucination and parse errors
- Few-shot examples are used for the quiz/flashcard generators
- Edge cases (empty context, off-topic queries) are handled explicitly

Credits:
- Prompt structure informed by Anthropic's prompting guide
  (https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview)
- JSON-first output discipline influenced by OpenAI's structured outputs
  (https://platform.openai.com/docs/guides/structured-outputs)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ---------- System prompts ----------
SYSTEM_QA = """You are StudyMate, a patient and accurate tutor.

Rules you MUST follow:
1. Answer ONLY using the information in the <context> block below. If the
   context does not contain the answer, say so clearly — do not invent facts.
2. Be concise but thorough. Use bullet points when listing items.
3. When quoting directly, keep quotes short (under 15 words) and cite the
   source name from the metadata.
4. If the user's question is ambiguous, ask a clarifying question instead
   of guessing.
5. Never discuss topics outside the provided context, even if asked."""

SYSTEM_QUIZ = """You are a quiz-writing expert for education. You produce
high-quality assessment items grounded in source material.

Your output MUST be valid JSON — a single JSON array — with no prose before
or after. Each element of the array is a question object with these fields:
- question (string)
- options (object with keys A, B, C, D — only for MCQ; omit for other types)
- answer (string — the correct option letter for MCQ, or the full text otherwise)
- explanation (string — why this answer is correct, grounded in the source)

Rules:
- Test understanding, not trivia
- Distractors should be plausible but clearly wrong
- Do not invent facts not present in the source"""

SYSTEM_FLASHCARDS = """You are a flashcard designer following spaced-repetition
best practices (minimum-information principle, atomic concepts).

Output MUST be a valid JSON array. Each card has:
- front (string — a focused prompt, usually a question or term)
- back (string — a single, atomic answer, 1-3 sentences max)

Rules:
- One concept per card
- Keep the front short (under 15 words)
- Do not duplicate concepts across cards"""

SYSTEM_SUMMARY = """You are a skilled academic writer. Produce summaries that
are faithful to the source, well-organized, and free of filler.

Rules:
- Do not introduce information not in the source
- Use your own wording; do not copy sentences verbatim
- Preserve technical precision"""

SYSTEM_ELI5 = """You are a gifted teacher who explains complex topics simply
without losing accuracy. Match the vocabulary and analogies to the audience
level requested."""


class PromptManager:
    """Builds prompts and routes calls to the selected LLM backend."""

    def __init__(self):
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

    # ---------- Prompt builders ----------
    def build_qa_prompt(self, query: str, context: str) -> dict:
        context = context.strip() or "(no relevant context found)"
        user = (
            f"<context>\n{context}\n</context>\n\n"
            f"<question>{query}</question>\n\n"
            "Answer the question using only the context above."
        )
        return {"system": SYSTEM_QA, "user": user}

    def build_quiz_prompt(
        self,
        context: str,
        num_questions: int,
        question_type: str = "MCQ",
        difficulty: str = "Medium",
    ) -> dict:
        example = self._quiz_few_shot(question_type)
        user = (
            f"<source>\n{context}\n</source>\n\n"
            f"Generate exactly {num_questions} {question_type} question(s) at "
            f"{difficulty} difficulty. Return JSON only.\n\n"
            f"Example of the output format:\n{example}"
        )
        return {"system": SYSTEM_QUIZ, "user": user}

    def build_flashcard_prompt(self, context: str, num_cards: int) -> dict:
        example = '[{"front": "What is photosynthesis?", ' \
                  '"back": "The process by which plants convert light energy into chemical energy stored in glucose."}]'
        user = (
            f"<source>\n{context}\n</source>\n\n"
            f"Generate exactly {num_cards} flashcards. Return JSON only.\n\n"
            f"Format example: {example}"
        )
        return {"system": SYSTEM_FLASHCARDS, "user": user}

    def build_summary_prompt(self, context: str, length: str) -> dict:
        instructions = {
            "One-paragraph": "Write a single flowing paragraph of 4-6 sentences.",
            "Bullet points (5-10)": "Produce 5 to 10 bullet points covering key ideas.",
            "Detailed outline": "Produce a hierarchical outline with sections and sub-bullets.",
            "Executive summary": "Produce a 3-part executive summary: Context, Key Findings, Implications.",
        }.get(length, "Produce a concise summary.")
        user = f"<source>\n{context}\n</source>\n\n{instructions}"
        return {"system": SYSTEM_SUMMARY, "user": user}

    def build_eli5_prompt(self, concept: str, context: str, audience: str) -> dict:
        user = (
            f"<source>\n{context}\n</source>\n\n"
            f"Explain the concept of **{concept}** for a {audience}. "
            "Use analogies appropriate to their level. Ground the explanation "
            "in the source where possible, but you may extend with general knowledge "
            "to build intuition."
        )
        return {"system": SYSTEM_ELI5, "user": user}

    @staticmethod
    def _quiz_few_shot(q_type: str) -> str:
        if q_type.upper() == "MCQ":
            return json.dumps([{
                "question": "What is the primary function of mitochondria?",
                "options": {
                    "A": "Protein synthesis",
                    "B": "ATP production via cellular respiration",
                    "C": "DNA storage",
                    "D": "Lipid digestion",
                },
                "answer": "B",
                "explanation": "Mitochondria are the site of oxidative phosphorylation, which generates ATP.",
            }], indent=2)
        if q_type.lower().startswith("true"):
            return json.dumps([{
                "question": "Mitochondria contain their own DNA.",
                "answer": "True",
                "explanation": "Mitochondria have a small circular genome inherited maternally.",
            }], indent=2)
        return json.dumps([{
            "question": "Briefly explain the function of mitochondria.",
            "answer": "They generate ATP through cellular respiration.",
            "explanation": "This is the primary energy-production role covered in the source.",
        }], indent=2)

    # ---------- LLM calling ----------
    def call_llm(self, prompt: dict, model: str = "claude-sonnet-4-5",
                 max_tokens: int = 1500, temperature: float = 0.3) -> str:
        """Route the prompt to the appropriate backend."""
        system = prompt["system"]
        user = prompt["user"]

        try:
            if model.startswith("claude"):
                return self._call_anthropic(system, user, model, max_tokens, temperature)
            if model.startswith("gpt"):
                return self._call_openai(system, user, model, max_tokens, temperature)
            # local / offline fallback
            return self._call_local(system, user)
        except Exception as e:
            logger.exception("LLM call failed")
            return f"⚠️ LLM call failed: {e}"

    def _call_anthropic(self, system, user, model, max_tokens, temperature) -> str:
        try:
            from anthropic import Anthropic
        except ImportError:
            return "⚠️ anthropic package not installed. Run: pip install anthropic"
        if not self.anthropic_key:
            return "⚠️ ANTHROPIC_API_KEY not set in environment."
        client = Anthropic(api_key=self.anthropic_key)
        # Map friendly names to current API model strings
        model_map = {
            "claude-sonnet-4-5": "claude-sonnet-4-5",
            "claude-haiku-4-5": "claude-haiku-4-5-20251001",
        }
        resp = client.messages.create(
            model=model_map.get(model, model),
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    def _call_openai(self, system, user, model, max_tokens, temperature) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            return "⚠️ openai package not installed. Run: pip install openai"
        if not self.openai_key:
            return "⚠️ OPENAI_API_KEY not set in environment."
        client = OpenAI(api_key=self.openai_key)
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    def _call_local(self, system, user) -> str:
        """Offline stub for demo/test environments without API keys."""
        return (
            "⚠️ No API key configured — returning a stubbed response.\n\n"
            f"[system prompt length: {len(system)}; user prompt length: {len(user)}]\n"
            "Configure ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env to enable real answers."
        )

    @staticmethod
    def extract_json(text: str):
        """
        Robust JSON extraction from LLM output.
        Handles responses wrapped in markdown code fences or containing
        preamble/postamble text.
        """
        if not text:
            return None
        # Strip code fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Find first balanced JSON array or object
        for open_c, close_c in [("[", "]"), ("{", "}")]:
            start = text.find(open_c)
            if start == -1:
                continue
            depth = 0
            for i in range(start, len(text)):
                if text[i] == open_c:
                    depth += 1
                elif text[i] == close_c:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
        logger.warning("Failed to extract JSON from LLM response")
        return None
