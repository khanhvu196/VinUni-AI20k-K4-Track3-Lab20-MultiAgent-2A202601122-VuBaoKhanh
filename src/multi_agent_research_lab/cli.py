"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import (
    configure_langfuse,
    flush_traces,
    trace_span,
)
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_langfuse(settings)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _run_baseline(query: str) -> ResearchState:
    """Run the reusable single-agent baseline for CLI and benchmarks."""

    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    with trace_span("single_agent_baseline", {"query": request.query}) as span:
        response = LLMClient().complete(
            system_prompt=(
                "You are a single-agent research assistant. Answer the user's research question "
                "clearly and concisely. Structure the response with findings, limitations, and a "
                "short conclusion. Do not invent citations or claim to have browsed the web."
            ),
            user_prompt=(
                f"Research question: {request.query}\n"
                f"Target audience: {request.audience}\n"
                "Complete the research task independently in one response."
            ),
        )
        span["attributes"].update(
            {
                "model": get_settings().gemini_model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            }
        )
    state.final_answer = response.content
    state.add_trace_event(
        "baseline_llm_call",
        {
            "model": get_settings().gemini_model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        },
    )
    return state


def _run_multi_agent(query: str) -> ResearchState:
    """Run the reusable multi-agent workflow for CLI and benchmarks."""

    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow().run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run one Gemini call as the single-agent baseline."""

    _init()
    _parse_query(query)
    state = _run_baseline(query)
    console.print(
        Panel.fit(state.final_answer or "No answer generated", title="Single-Agent Baseline")
    )
    flush_traces()


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent research workflow."""

    _init()
    _parse_query(query)
    result = _run_multi_agent(query)
    console.print(result.model_dump_json(indent=2))
    flush_traces()


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Benchmark query")] = (
        "When should teams use single-agent versus multi-agent architectures for complex "
        "research tasks?"
    ),
) -> None:
    """Compare baseline and multi-agent runs and save report artifacts."""

    _init()
    _parse_query(query)
    baseline_state, baseline_metrics = run_benchmark("single-agent", query, _run_baseline)
    multi_state, multi_metrics = run_benchmark("multi-agent", query, _run_multi_agent)
    report = render_markdown_report([baseline_metrics, multi_metrics])
    store = LocalArtifactStore()
    report_path = store.write_text("benchmark_report.md", report)
    store.write_text("baseline_trace.json", baseline_state.model_dump_json(indent=2))
    store.write_text("multi_agent_trace.json", multi_state.model_dump_json(indent=2))
    console.print(report)
    console.print(f"Saved benchmark report to {report_path}")
    flush_traces()


if __name__ == "__main__":
    app()
