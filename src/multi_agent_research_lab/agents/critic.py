"""Critic agent for final-answer evidence and citation auditing."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Audit the final answer against retrieved evidence and append findings."""

        if not state.final_answer or not state.sources:
            raise AgentExecutionError("Critic requires a final answer and sources")

        source_ledger = "\n".join(
            f"- [{source.metadata.get('source_id', 'unknown')}] {source.title}; "
            f"synthetic={bool(source.metadata.get('is_synthetic', False))}; "
            f"evidence={source.snippet[:600]}"
            for source in state.sources
        )
        response = self.llm_client.complete(
            system_prompt=(
                "You are an independent Critic. Audit the final answer against the supplied source "
                "ledger. Check whether major claims have supporting [source_id] citations, "
                "identify unsupported or overstated claims, verify synthetic evidence is labeled, "
                "and assess "
                "hallucination risk. Return: Verdict (PASS or REVISE), Citation audit, Unsupported "
                "claims, and Concrete revision recommendations. Do not rewrite the answer."
            ),
            user_prompt=(
                f"Research question: {state.request.query}\n\n"
                f"Source ledger:\n{source_ledger}\n\n"
                f"Final answer:\n{state.final_answer}"
            ),
        )

        state.critic_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "critic_complete",
            {"output_tokens": response.output_tokens},
        )
        return state
