from multi_agent_research_lab.agents import CriticAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse


class FakeLLMClient(LLMClient):
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        assert "Final answer" in user_prompt
        assert "[A01]" in user_prompt
        return LLMResponse(content="Verdict: PASS\nCitation audit: All claims supported. [A01]")


def test_critic_populates_audit_and_result() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[
            SourceDocument(
                title="Evidence",
                snippet="Role specialization.",
                metadata={"source_id": "A01", "is_synthetic": False},
            )
        ],
        research_notes="Research notes. [A01]",
        analysis_notes="Analysis notes. [A01]",
        final_answer="Final answer. [A01]",
    )

    result = CriticAgent(FakeLLMClient()).run(state)

    assert result.critic_notes.startswith("Verdict: PASS")
    assert result.agent_results[-1].agent == AgentName.CRITIC
    assert result.trace[-1]["name"] == "critic_complete"
