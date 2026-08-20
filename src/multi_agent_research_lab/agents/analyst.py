"""Analyst agent for evidence comparison and uncertainty analysis."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Compare cited claims, assess evidence quality, and update shared state."""

        if not state.research_notes or not state.sources:
            raise AgentExecutionError("Analyst requires research notes and sources")

        source_ledger = "\n".join(
            f"- [{source.metadata.get('source_id', 'unknown')}]: "
            f"class={source.metadata.get('document_class', 'unknown')}, "
            f"synthetic={bool(source.metadata.get('is_synthetic', False))}"
            for source in state.sources
        )
        response = self.llm_client.complete(
            system_prompt=(
                "You are the Analyst in a multi-agent research system. Compare the research "
                "claims, "
                "identify agreement and conflict, assess source strength, flag weak or synthetic "
                "evidence, and state uncertainty. Preserve exact [source_id] citations. Do not add "
                "new factual claims. Produce structured analysis for a Writer, not a final answer."
            ),
            user_prompt=(
                f"Research question: {state.request.query}\n\n"
                f"Source ledger:\n{source_ledger}\n\n"
                f"Research notes:\n{state.research_notes}"
            ),
        )

        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "analyst_complete",
            {"output_tokens": response.output_tokens},
        )
        return state
