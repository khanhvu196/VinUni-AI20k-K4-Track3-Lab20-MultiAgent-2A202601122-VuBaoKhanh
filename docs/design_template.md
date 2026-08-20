# Multi-Agent Research System Design

## Problem

Hệ thống nhận câu hỏi nghiên cứu về AI agents, truy xuất evidence từ corpus offline,
phân tích độ tin cậy và viết câu trả lời có citation. Bài so sánh một Gemini call duy nhất
với workflow Supervisor → Researcher → Analyst → Writer dùng cùng model Gemini 3.6 Flash.

## Why multi-agent?

Single-agent nhanh và rẻ nhưng phải giữ toàn bộ planning, evidence và writing trong một
context. Multi-agent tách retrieval, evidence analysis và synthesis thành artifacts độc lập,
giúp trace/debug và kiểm soát citation rõ hơn. Benchmark thực tế dùng để xác định lợi ích này
có xứng đáng với latency/token overhead hay không, thay vì mặc định multi-agent tốt hơn.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Route theo artifact còn thiếu và dừng đúng lúc | Shared state | Route kế tiếp | Loop hoặc route không hợp lệ |
| Researcher | Truy xuất và trích notes có citation | Query, max_sources | Sources, research_notes | Retrieval kém hoặc LLM lỗi |
| Analyst | So sánh claims, source quality và uncertainty | Sources, research_notes | analysis_notes | Khuếch đại evidence yếu |
| Writer | Viết câu trả lời có citation và limitations | Sources, analysis_notes | final_answer | Bịa nguồn hoặc bỏ uncertainty |
| Critic | Audit citation, evidence và hallucination risk | Sources, final_answer | critic_notes | Bỏ sót unsupported claim |

## Shared state

- `request`: query, audience và giới hạn sources đã được Pydantic validate.
- `iteration`, `route_history`: giải thích routing và chặn loop.
- `sources`: evidence cùng source ID, provenance và synthetic flag.
- `research_notes`, `analysis_notes`, `final_answer`: artifact riêng của từng handoff.
- `critic_notes`: verdict và đề xuất sửa từ bước audit độc lập.
- `agent_results`: structured result cùng token/cost metadata.
- `trace`, `errors`: local audit trail, latency spans và fallback diagnostics.

## Routing policy

```text
START → Supervisor
          ├─ thiếu sources/research_notes → Researcher ─┐
          ├─ thiếu analysis_notes         → Analyst ────┤
          ├─ thiếu final_answer           → Writer ─────┤
          ├─ thiếu critic_notes           → Critic ─────┤
          └─ đủ output/đạt iteration limit → END        │
                    Supervisor ←─────────────────────────┘
```

## Guardrails

- Max iterations: mặc định 6, cấu hình bằng `MAX_ITERATIONS`.
- Timeout: Gemini request mặc định 60 giây qua `TIMEOUT_SECONDS`.
- Retry: tối đa 3 provider attempts với exponential backoff 1–4 giây.
- Fallback: Researcher dùng cited snippets; Analyst dùng research notes; Writer dùng analysis;
  Critic yêu cầu manual citation review.
- Validation: Pydantic input/state, allowlist route và LangGraph recursion limit.

## Benchmark plan

Query chính: “When should teams use single-agent versus multi-agent architectures for
complex research tasks?” Cả hai chạy cùng Gemini model. Metrics: latency, total tokens,
free-tier cost, quality proxy 0–10, citation coverage và failure rate.

Kết quả thực tế:

| Run | Latency | Tokens | Quality proxy | Citation coverage | Failure rate |
|---|---:|---:|---:|---:|---:|
| Single-agent | 22.70s | 1,198 | 10.0 | N/A | 0% |
| Multi-agent + Critic | 110.51s | 9,060 | 10.0 | 100% | 0% |

Multi-agent không tăng quality proxy trong lần chạy này nhưng cung cấp evidence grounding,
100% citation coverage, Critic audit và trace theo vai trò; đổi lại chậm hơn 4.9× và dùng
token nhiều hơn 7.6×.
