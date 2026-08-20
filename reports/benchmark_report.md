# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| single-agent | 22.70 | 0.0000 | 10.0 |  | 0% | tokens=1198; routes=0 |
| multi-agent | 110.51 | 0.0000 | 10.0 | 100% | 0% | tokens=9060; routes=5 |

## Methodology

Both systems use the same Gemini model and query. Latency is end-to-end wall-clock time. Token totals come from provider usage metadata. Cost is USD 0 for the configured Gemini free tier. Citation coverage is the fraction of retrieved source IDs cited in the final answer; it is not applicable when a run retrieves no sources.

Quality is a reproducible 0–10 proxy: non-empty answer, at least 100 words, and explicit Findings, Limitations, and Conclusion sections are each worth two points. A peer reviewer should replace this proxy with the course rubric score for final submission.

## Failure mode and mitigation

A weak or unsupported Researcher note can cascade through Analyst and Writer. The system
mitigates this by preserving source IDs and synthetic labels in shared state, requiring
citations at every handoff, bounding iterations, validating routes, applying provider
timeouts/retries, running a final Critic audit, and exposing each agent span in Langfuse.

## Trace evidence

The screenshot below captures the core Supervisor → Researcher → Analyst → Writer workflow.
The subsequent bonus benchmark recorded `researcher → analyst → writer → critic → done` in
`route_history`, produced non-empty `critic_notes`, and exported the additional Critic span to
the same Langfuse project.

![Langfuse multi-agent trace](langfuse_multi_agent_trace.png)

## Peer review

**Reviewer:** Võ Hồ Nhật Nam — 2A202601700

**Strength:** Các agent có vai trò rõ, shared state giữ được source ID và trace đầy đủ.

**Risk / failure mode:** Nếu Researcher lấy evidence chưa phù hợp thì lỗi có thể truyền sang
Analyst và Writer.

**One concrete improvement:** Critic đã được thêm để kiểm tra citation và chất lượng evidence
trước khi trả final answer.

**Score:** 9/10
