"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render metrics with methodology and the required failure-mode analysis."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )
    lines.extend(
        [
            "",
            "## Methodology",
            "",
            "Both systems use the same Gemini model and query. Latency is end-to-end wall-clock "
            "time. Token totals come from provider usage metadata. Cost is USD 0 for the "
            "configured "
            "Gemini free tier. Citation coverage is the fraction of retrieved source IDs cited in "
            "the final answer; it is not applicable when a run retrieves no sources.",
            "",
            "Quality is a reproducible 0–10 proxy: non-empty answer, at least 100 words, "
            "and explicit Findings, Limitations, and Conclusion sections are each worth two "
            "points. A peer reviewer "
            "should replace this proxy with the course rubric score for final submission.",
            "",
            "## Failure mode and mitigation",
            "",
            "A weak or unsupported Researcher note can cascade through Analyst and Writer. "
            "The system mitigates this by preserving source IDs and synthetic labels in shared "
            "state, requiring "
            "citations at every handoff, bounding iterations, validating routes, applying provider "
            "timeouts/retries, running a final Critic audit, and exposing each agent span in "
            "Langfuse for audit.",
            "",
            "## Trace evidence",
            "",
            "Open the configured Langfuse project and capture the multi-agent trace showing "
            "Supervisor, Researcher, Analyst, Writer, and Critic spans.",
        ]
    )
    return "\n".join(lines) + "\n"
