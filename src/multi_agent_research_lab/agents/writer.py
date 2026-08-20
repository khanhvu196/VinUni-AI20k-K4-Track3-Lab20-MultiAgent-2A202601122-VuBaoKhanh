"""Writer agent for cited final-answer synthesis."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Synthesize the final answer with citations and explicit limitations."""

        if not state.analysis_notes or not state.sources:
            raise AgentExecutionError("Writer requires analysis notes and sources")

        source_ledger = "\n".join(
            f"- [{source.metadata.get('source_id', 'unknown')}] {source.title}; "
            f"synthetic={bool(source.metadata.get('is_synthetic', False))}"
            for source in state.sources
        )
        response = self.llm_client.complete(
            system_prompt=(
                "You are the Writer in a multi-agent research system. Write a clear answer for "
                "the requested audience using only the supplied analysis and source ledger. Cite "
                "major claims with exact [source_id] markers. Clearly label synthetic evidence, "
                "preserve uncertainty, and include Findings, Limitations, and Conclusion sections. "
                "Do not invent references."
            ),
            user_prompt=(
                f"Research question: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Source ledger:\n{source_ledger}\n\n"
                f"Analysis notes:\n{state.analysis_notes}"
            ),
        )

        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "writer_complete",
            {"output_tokens": response.output_tokens},
        )
        return state
