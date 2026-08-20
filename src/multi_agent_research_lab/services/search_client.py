"""Offline corpus retrieval for ResearcherAgent."""

import json
import re
from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Rank evidence from the self-contained benchmark corpus."""

    def __init__(self, corpus_root: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self.corpus_root = (
            corpus_root or project_root / "ai_agent_offline_research_corpus_v2/topics"
        )

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    @classmethod
    def _score(cls, query_terms: set[str], text: str, multiplier: int = 1) -> int:
        return len(query_terms & cls._terms(text)) * multiplier

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return the most relevant articles and documents with citation metadata."""

        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        if not self.corpus_root.is_dir():
            raise AgentExecutionError(f"Offline corpus not found: {self.corpus_root}")

        query_terms = self._terms(query)
        ranked: list[tuple[int, SourceDocument]] = []
        for path in sorted(self.corpus_root.glob("*.json")):
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            metadata = data.get("benchmark_metadata", {})
            topic = data.get("topic", {})
            topic_text = " ".join(
                [
                    str(topic.get("name", "")),
                    str(topic.get("research_question", "")),
                    " ".join(topic.get("tags", [])),
                ]
            )
            topic_score = self._score(query_terms, topic_text, multiplier=10)
            knowledge = data.get("knowledge_base", {})

            for article in knowledge.get("knowledge_articles", []):
                title = str(article.get("title", "Untitled article"))
                content = str(article.get("content", ""))
                score = topic_score + self._score(query_terms, title, 3)
                score += self._score(query_terms, content)
                ranked.append(
                    (
                        score,
                        SourceDocument(
                            title=title,
                            snippet=content[:2000],
                            metadata={
                                "source_id": article.get("article_id"),
                                "topic_id": metadata.get("topic_id"),
                                "document_class": "knowledge_article",
                                "is_synthetic": False,
                                "corpus_file": path.name,
                            },
                        ),
                    )
                )

            for document in knowledge.get("source_documents", []):
                title = str(document.get("title", "Untitled document"))
                content = str(document.get("full_text", ""))
                score = topic_score + self._score(query_terms, title, 3)
                score += self._score(query_terms, content)
                ranked.append(
                    (
                        score,
                        SourceDocument(
                            title=title,
                            url=document.get("provenance_url"),
                            snippet=content[:2000],
                            metadata={
                                "source_id": document.get("document_id"),
                                "topic_id": metadata.get("topic_id"),
                                "document_class": document.get("document_class"),
                                "is_synthetic": bool(document.get("is_synthetic", False)),
                                "recommended_weight": document.get("recommended_weight"),
                                "corpus_file": path.name,
                            },
                        ),
                    )
                )

        if not ranked:
            raise AgentExecutionError("Offline corpus contains no searchable documents")
        ranked.sort(key=lambda item: (item[0], item[1].title), reverse=True)
        return [document for _, document in ranked[:max_results]]
