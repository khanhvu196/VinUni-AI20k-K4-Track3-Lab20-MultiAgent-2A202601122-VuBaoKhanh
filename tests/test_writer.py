from multi_agent_research_lab.agents import WriterAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse


class FakeLLMClient(LLMClient):
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        assert "Analysis notes" in user_prompt
        assert "[A01]" in user_prompt
        return LLMResponse(
            content="# Findings\nRole clarity helps. [A01]\n# Limitations\nLimited evidence."
        )


def test_writer_populates_final_answer_and_result() -> None:
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
    )

    result = WriterAgent(FakeLLMClient()).run(state)

    assert result.final_answer
    assert "[A01]" in result.final_answer
    assert result.agent_results[-1].agent == AgentName.WRITER
    assert result.trace[-1]["name"] == "writer_complete"
