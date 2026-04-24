"""
Synthetic Data Generator Module
-------------------------------
Generates learning artifacts (quizzes, flashcards, summaries, explanations)
from the RAG-indexed corpus.

This component addresses the "Synthetic Data Generation" requirement of
the assignment by using the LLM to produce:
- Practice quiz items (with answer keys and distractors)
- Spaced-repetition flashcards
- Multi-length summaries
- Level-adapted explanations

Diversity is encouraged by sampling different context windows from the
knowledge base rather than always using the same top-K chunks.

Credits:
- Distractor-quality heuristics informed by Haladyna et al. (2002),
  "A Review of Multiple-Choice Item-Writing Guidelines"
- Spaced-repetition principles from Wozniak (SuperMemo) and Piotr Wozniak's
  "Effective learning: Twenty rules of formulating knowledge"
"""

from __future__ import annotations

import logging
import random
from typing import List, Dict, Optional

from .prompt_templates import PromptManager
from .rag_engine import RAGEngine

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    def __init__(self, rag_engine: RAGEngine, prompt_manager: PromptManager, model: str = "claude-sonnet-4-5"):
        self.rag = rag_engine
        self.pm = prompt_manager
        self.model = model

    # -------------------- Context gathering --------------------
    def _gather_context(self, topic: Optional[str], max_chars: int = 6000) -> str:
        """
        Retrieve context for a generation task.

        - If a topic is given, use RAG retrieval.
        - If no topic, sample diverse chunks from the collection so the
          output isn't biased to the first section of the document.
        """
        if topic:
            hits = self.rag.query(topic, top_k=6)
            return self._concat_within_limit(hits, max_chars)

        # No topic → sample diversely. ChromaDB's peek gives a contiguous
        # slice; we shuffle afterwards to spread coverage.
        try:
            peek = self.rag.collection.peek(limit=50)
            docs = peek.get("documents", [])
            metas = peek.get("metadatas", [])
            items = list(zip(docs, metas))
            random.shuffle(items)
            hits = [{"text": d, "metadata": m} for d, m in items]
        except Exception as e:
            logger.warning(f"peek() failed, falling back to empty context: {e}")
            hits = []
        return self._concat_within_limit(hits, max_chars)

    @staticmethod
    def _concat_within_limit(hits: List[Dict], max_chars: int) -> str:
        buf, used = [], 0
        for h in hits:
            t = h.get("text", "")
            if used + len(t) > max_chars:
                break
            buf.append(t)
            used += len(t)
        return "\n\n".join(buf)

    # -------------------- Public API --------------------
    def generate_quiz(
        self,
        topic: Optional[str] = None,
        num_questions: int = 5,
        question_type: str = "MCQ",
        difficulty: str = "Medium",
    ) -> List[Dict]:
        context = self._gather_context(topic)
        if not context:
            logger.warning("No context available for quiz generation")
            return []

        prompt = self.pm.build_quiz_prompt(
            context=context,
            num_questions=num_questions,
            question_type=question_type,
            difficulty=difficulty,
        )
        raw = self.pm.call_llm(prompt, model=self.model, temperature=0.5)
        parsed = self.pm.extract_json(raw)
        if not isinstance(parsed, list):
            logger.warning("Quiz output was not a list")
            return []
        return self._validate_quiz(parsed, question_type)

    def generate_flashcards(
        self, topic: Optional[str] = None, num_cards: int = 10
    ) -> List[Dict]:
        context = self._gather_context(topic)
        if not context:
            return []
        prompt = self.pm.build_flashcard_prompt(context=context, num_cards=num_cards)
        raw = self.pm.call_llm(prompt, model=self.model, temperature=0.4)
        parsed = self.pm.extract_json(raw)
        if not isinstance(parsed, list):
            return []
        # Basic validation
        return [c for c in parsed if isinstance(c, dict) and c.get("front") and c.get("back")]

    def generate_summary(
        self, topic: Optional[str] = None, length: str = "Bullet points (5-10)"
    ) -> str:
        context = self._gather_context(topic, max_chars=8000)
        if not context:
            return "No content available to summarize."
        prompt = self.pm.build_summary_prompt(context=context, length=length)
        return self.pm.call_llm(prompt, model=self.model, temperature=0.3)

    def generate_eli5(self, concept: str, audience: str = "5-year-old") -> str:
        context = self._gather_context(concept, max_chars=4000)
        prompt = self.pm.build_eli5_prompt(
            concept=concept, context=context, audience=audience
        )
        return self.pm.call_llm(prompt, model=self.model, temperature=0.7)

    # -------------------- Validation helpers --------------------
    @staticmethod
    def _validate_quiz(items: List[Dict], q_type: str) -> List[Dict]:
        """Filter malformed quiz items and normalize keys."""
        valid = []
        for q in items:
            if not isinstance(q, dict) or not q.get("question"):
                continue
            if q_type.upper() == "MCQ":
                opts = q.get("options")
                if not isinstance(opts, dict) or len(opts) < 2:
                    continue
                if not q.get("answer"):
                    continue
            valid.append(q)
        return valid
