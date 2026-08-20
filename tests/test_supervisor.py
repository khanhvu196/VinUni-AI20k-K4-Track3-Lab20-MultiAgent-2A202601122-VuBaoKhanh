from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_supervisor_routes_through_missing_artifacts() -> None:
    state = _state()
    supervisor = SupervisorAgent(max_iterations=6)

    supervisor.run(state)
    assert state.route_history[-1] == "researcher"

    state.sources = [SourceDocument(title="Source", snippet="Evidence")]
    state.research_notes = "Research notes"
    supervisor.run(state)
    assert state.route_history[-1] == "analyst"

    state.analysis_notes = "Analysis notes"
    supervisor.run(state)
    assert state.route_history[-1] == "writer"

    state.final_answer = "Final answer"
    supervisor.run(state)
    assert state.route_history[-1] == "critic"

    state.critic_notes = "Critic audit"
    supervisor.run(state)
    assert state.route_history[-1] == "done"
    assert state.trace[-1]["name"] == "supervisor_route"


def test_supervisor_stops_at_iteration_limit() -> None:
    state = _state()
    state.iteration = 2

    SupervisorAgent(max_iterations=2).run(state)

    assert state.route_history == ["done"]
    assert state.errors == ["Maximum workflow iterations reached"]
