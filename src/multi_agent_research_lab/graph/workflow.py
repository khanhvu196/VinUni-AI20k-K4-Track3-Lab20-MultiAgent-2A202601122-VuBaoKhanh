"""LangGraph orchestration for the multi-agent research workflow."""

from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.supervisor = SupervisorAgent(max_iterations=self.settings.max_iterations)
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()

    @staticmethod
    def _run_agent(agent: BaseAgent, state: ResearchState) -> dict[str, Any]:
        """Run an agent and convert its Pydantic state to a LangGraph update."""

        previous_result_count = len(state.agent_results)
        with trace_span(
            agent.name,
            {"iteration": state.iteration, "query": state.request.query},
        ) as span:
            try:
                result = agent.run(state)
            except Exception as exc:
                state.errors.append(f"{agent.name} failed after retries: {exc}")
                span["attributes"].update({"fallback": True, "error": str(exc)})
                if agent.name == "researcher" and state.sources:
                    state.research_notes = "\n".join(
                        f"- [{source.metadata.get('source_id', 'unknown')}] {source.snippet}"
                        for source in state.sources
                    )
                elif agent.name == "analyst" and state.research_notes:
                    state.analysis_notes = "Provisional analysis fallback:\n" + state.research_notes
                elif agent.name == "writer" and (state.analysis_notes or state.research_notes):
                    state.final_answer = state.analysis_notes or state.research_notes
                elif agent.name == "critic" and state.final_answer:
                    state.critic_notes = "Critic unavailable; manual citation review required."
                else:
                    raise
                result = state
            if len(result.agent_results) > previous_result_count:
                span["attributes"].update(result.agent_results[-1].metadata)
        result.add_trace_event("agent_span", span)
        return result.model_dump()

    def _run_supervisor(self, state: ResearchState) -> dict[str, Any]:
        return self._run_agent(self.supervisor, state)

    def _run_researcher(self, state: ResearchState) -> dict[str, Any]:
        return self._run_agent(self.researcher, state)

    def _run_analyst(self, state: ResearchState) -> dict[str, Any]:
        return self._run_agent(self.analyst, state)

    def _run_writer(self, state: ResearchState) -> dict[str, Any]:
        return self._run_agent(self.writer, state)

    def _run_critic(self, state: ResearchState) -> dict[str, Any]:
        return self._run_agent(self.critic, state)

    @staticmethod
    def _next_route(
        state: ResearchState,
    ) -> Literal["researcher", "analyst", "writer", "critic", "done"]:
        """Read the route just recorded by Supervisor."""

        if not state.route_history:
            raise ValueError("Supervisor did not record a route")
        route = state.route_history[-1]
        if route not in {"researcher", "analyst", "writer", "critic", "done"}:
            raise ValueError(f"Supervisor returned invalid route: {route}")
        return cast(Literal["researcher", "analyst", "writer", "critic", "done"], route)

    def build(self) -> CompiledStateGraph[ResearchState, None, ResearchState, ResearchState]:
        """Create and compile nodes, return edges, conditional routes, and stop edge.

        Each worker writes its artifact to shared state and returns control to Supervisor.
        """

        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", self._run_supervisor)
        graph.add_node("researcher", self._run_researcher)
        graph.add_node("analyst", self._run_analyst)
        graph.add_node("writer", self._run_writer)
        graph.add_node("critic", self._run_critic)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._next_route,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                "done": END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        graph.add_edge("critic", "supervisor")
        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Invoke the graph with a bounded recursion limit and validate final state."""

        result = self.build().invoke(
            state,
            config={"recursion_limit": self.settings.max_iterations * 2 + 2},
        )
        return ResearchState.model_validate(result)
