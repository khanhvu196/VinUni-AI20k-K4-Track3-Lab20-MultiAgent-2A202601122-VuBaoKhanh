from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


class FailingWriter(BaseAgent):
    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        raise RuntimeError("provider unavailable")


def test_writer_failure_falls_back_to_analysis() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[SourceDocument(title="Source", snippet="Evidence")],
        research_notes="Research notes",
        analysis_notes="Analysis notes",
    )
    workflow = MultiAgentWorkflow()
    workflow.writer = FailingWriter()
    workflow.supervisor.max_iterations = 1

    result = workflow.run(state)

    assert result.final_answer == "Analysis notes"
    assert result.route_history == ["writer", "done"]
    assert "writer failed after retries" in result.errors[0]
