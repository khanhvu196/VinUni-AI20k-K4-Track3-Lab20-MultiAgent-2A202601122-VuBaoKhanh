"""Benchmark skeleton for single-agent vs multi-agent."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, usage, output quality proxy, citations, and failures."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    answer = state.final_answer or ""
    costs: list[float] = []
    for agent_result in state.agent_results:
        cost = agent_result.metadata.get("cost_usd")
        if isinstance(cost, int | float) and not isinstance(cost, bool):
            costs.append(float(cost))
    input_tokens = sum(
        int(result.metadata.get("input_tokens") or 0) for result in state.agent_results
    )
    output_tokens = sum(
        int(result.metadata.get("output_tokens") or 0) for result in state.agent_results
    )
    for event in state.trace:
        if event.get("name") != "baseline_llm_call":
            continue
        payload = event.get("payload", {})
        input_tokens += int(payload.get("input_tokens") or 0)
        output_tokens += int(payload.get("output_tokens") or 0)
        baseline_cost = payload.get("cost_usd")
        if isinstance(baseline_cost, int | float) and not isinstance(baseline_cost, bool):
            costs.append(float(baseline_cost))
    source_ids = [
        str(source.metadata["source_id"])
        for source in state.sources
        if source.metadata.get("source_id")
    ]
    citation_coverage = (
        sum(f"[{source_id}]" in answer for source_id in source_ids) / len(source_ids)
        if source_ids
        else None
    )
    quality_checks = [
        bool(answer.strip()),
        len(answer.split()) >= 100,
        "findings" in answer.lower(),
        "limitation" in answer.lower(),
        "conclusion" in answer.lower(),
    ]
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=sum(costs),
        quality_score=2.0 * sum(quality_checks),
        citation_coverage=citation_coverage,
        failure_rate=1.0 if state.errors or not answer.strip() else 0.0,
        notes=f"tokens={input_tokens + output_tokens}; routes={len(state.route_history)}",
    )
    return state, metrics
