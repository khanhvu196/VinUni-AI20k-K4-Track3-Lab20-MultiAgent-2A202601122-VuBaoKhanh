# Exit Ticket

## Khi nào nên dùng multi-agent?

Nên dùng khi nhiệm vụ nghiên cứu dài cần retrieval, đánh giá evidence, citation provenance và
debug từng handoff. Trong benchmark này, multi-agent đạt 100% citation coverage và cung cấp
trace riêng cho Supervisor, Researcher, Analyst và Writer. Lợi ích đó phù hợp với báo cáo có
giá trị cao, nơi khả năng audit quan trọng hơn tốc độ.

## Khi nào không nên dùng multi-agent?

Không nên dùng cho câu hỏi hẹp mà một model có thể trả lời trực tiếp và không cần evidence
audit. Benchmark cho thấy hai cách cùng đạt quality proxy 10.0, nhưng multi-agent mất 86.26
giây và 6,146 tokens so với 26.73 giây và 1,159 tokens của single-agent. Trong trường hợp này,
single-agent đơn giản, nhanh và tiết kiệm quota hơn.
