from multi_agent_research_lab.agents import ResearcherAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient


class FakeSearchClient(SearchClient):
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        return [
            SourceDocument(
                title="Evidence",
                snippet="Multi-agent systems separate responsibilities.",
                metadata={"source_id": "A01", "is_synthetic": False},
            )
        ]


class FakeLLMClient(LLMClient):
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        assert "[A01]" in user_prompt
        return LLMResponse(content="Role specialization improves clarity. [A01]", output_tokens=8)


def test_researcher_populates_sources_notes_and_result() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    agent = ResearcherAgent(FakeSearchClient(), FakeLLMClient())

    result = agent.run(state)

    assert result.sources[0].metadata["source_id"] == "A01"
    assert result.research_notes == "Role specialization improves clarity. [A01]"
    assert result.agent_results[-1].agent == AgentName.RESEARCHER
    assert result.trace[-1]["name"] == "researcher_complete"
