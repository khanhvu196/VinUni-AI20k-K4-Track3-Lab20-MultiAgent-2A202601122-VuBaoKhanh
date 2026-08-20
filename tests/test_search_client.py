from multi_agent_research_lab.services.search_client import SearchClient


def test_offline_search_returns_ranked_citation_metadata() -> None:
    results = SearchClient().search(
        "single-agent versus multi-agent research architecture",
        max_results=3,
    )

    assert len(results) == 3
    assert all(item.snippet for item in results)
    assert all(item.metadata.get("source_id") for item in results)
    assert all(item.metadata.get("topic_id") == "AIAGENT-01" for item in results)
