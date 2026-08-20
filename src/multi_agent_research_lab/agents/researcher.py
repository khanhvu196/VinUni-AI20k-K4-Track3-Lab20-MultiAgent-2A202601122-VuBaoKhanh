"""Researcher agent for evidence retrieval and note extraction."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Retrieve evidence, ask Gemini for cited notes, and update shared state."""

        sources = self.search_client.search(
            state.request.query,
            max_results=state.request.max_sources,
        )
        state.sources = sources
        evidence_blocks = []
        for source in sources:
            source_id = source.metadata.get("source_id", "unknown")
            synthetic = (
                "synthetic benchmark evidence"
                if source.metadata.get("is_synthetic")
                else "public reference summary"
            )
            evidence_blocks.append(
                f"[{source_id}] ({synthetic}) {source.title}\n{source.snippet[:1200]}"
            )

        response = self.llm_client.complete(
            system_prompt=(
                "You are the Researcher in a multi-agent system. Extract concise factual notes "
                "only from the supplied offline evidence. Cite every note with its exact "
                "[source_id]. "
                "Keep synthetic benchmark evidence explicitly labeled synthetic. Do not add facts "
                "from memory and list any unanswered part of the research question."
            ),
            user_prompt=(
                f"Research question: {state.request.query}\n\n"
                "Offline evidence:\n\n" + "\n\n".join(evidence_blocks)
            ),
        )

        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "source_count": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "researcher_complete",
            {"source_count": len(sources), "output_tokens": response.output_tokens},
        )
        return state
