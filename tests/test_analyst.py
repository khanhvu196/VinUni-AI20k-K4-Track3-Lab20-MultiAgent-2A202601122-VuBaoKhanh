from multi_agent_research_lab.agents import AnalystAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse


class FakeLLMClient(LLMClient):
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        assert "Research notes" in user_prompt
        assert "[A01]" in user_prompt
        return LLMResponse(content="The evidence supports role clarity with limitations. [A01]")


def test_analyst_populates_analysis_and_result() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[
            SourceDocument(
                title="Evidence",
                snippet="Role specialization.",
                metadata={"source_id": "A01", "is_synthetic": False},
            )
        ],
        research_notes="Role specialization improves clarity. [A01]",
    )

    result = AnalystAgent(FakeLLMClient()).run(state)

    assert result.analysis_notes
    assert result.agent_results[-1].agent == AgentName.ANALYST
    assert result.trace[-1]["name"] == "analyst_complete"
