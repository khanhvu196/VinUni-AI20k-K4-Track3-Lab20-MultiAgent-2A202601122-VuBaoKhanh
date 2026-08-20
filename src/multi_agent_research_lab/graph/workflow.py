"""LangGraph orchestration for the multi-agent research workflow."""

from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState


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

    @staticmethod
    def _run_agent(agent: BaseAgent, state: ResearchState) -> dict[str, Any]:
        """Run an agent and convert its Pydantic state to a LangGraph update."""

        return agent.run(state).model_dump()

    def _run_supervisor(self, state: ResearchState) -> dict[str, Any]:
        return self._run_agent(self.supervisor, state)

    def _run_researcher(self, state: ResearchState) -> dict[str, Any]:
        return self._run_agent(self.researcher, state)

    def _run_analyst(self, state: ResearchState) -> dict[str, Any]:
        return self._run_agent(self.analyst, state)

    def _run_writer(self, state: ResearchState) -> dict[str, Any]:
        return self._run_agent(self.writer, state)

    @staticmethod
    def _next_route(
        state: ResearchState,
    ) -> Literal["researcher", "analyst", "writer", "done"]:
        """Read the route just recorded by Supervisor."""

        if not state.route_history:
            raise ValueError("Supervisor did not record a route")
        route = state.route_history[-1]
        if route not in {"researcher", "analyst", "writer", "done"}:
            raise ValueError(f"Supervisor returned invalid route: {route}")
        return cast(Literal["researcher", "analyst", "writer", "done"], route)

    def build(self) -> CompiledStateGraph[ResearchState, None, ResearchState, ResearchState]:
        """Create and compile nodes, return edges, conditional routes, and stop edge.

        Each worker writes its artifact to shared state and returns control to Supervisor.
        """

        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", self._run_supervisor)
        graph.add_node("researcher", self._run_researcher)
        graph.add_node("analyst", self._run_analyst)
        graph.add_node("writer", self._run_writer)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._next_route,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Invoke the graph with a bounded recursion limit and validate final state."""

        result = self.build().invoke(
            state,
            config={"recursion_limit": self.settings.max_iterations * 2 + 2},
        )
        return ResearchState.model_validate(result)
