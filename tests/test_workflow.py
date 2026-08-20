from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_workflow_stops_when_state_is_complete() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[SourceDocument(title="Source", snippet="Evidence")],
        research_notes="Research notes",
        analysis_notes="Analysis notes",
        final_answer="Final answer",
        critic_notes="Critic audit",
    )

    result = MultiAgentWorkflow().run(state)

    assert result.final_answer == "Final answer"
    assert result.route_history == ["done"]
    assert [event["name"] for event in result.trace] == ["supervisor_route", "agent_span"]
