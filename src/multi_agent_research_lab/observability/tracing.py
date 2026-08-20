"""Local timing spans with optional Langfuse export for CLI runs."""

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from time import perf_counter
from typing import Any

from langfuse import Langfuse

from multi_agent_research_lab.core.config import Settings

_langfuse_client: Langfuse | None = None


def configure_langfuse(settings: Settings) -> bool:
    """Configure Langfuse when both project credentials are available."""

    global _langfuse_client
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        _langfuse_client = None
        return False
    _langfuse_client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_host,
        timeout=settings.timeout_seconds,
        environment=settings.app_env,
    )
    return True


def flush_traces() -> None:
    """Synchronously deliver buffered observations before a CLI process exits."""

    if _langfuse_client is not None:
        _langfuse_client.flush()


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Capture a serializable local span and mirror it to Langfuse when configured."""

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    remote_context = (
        _langfuse_client.start_as_current_observation(
            name=name,
            as_type="agent",
            input=span["attributes"],
        )
        if _langfuse_client is not None
        else nullcontext(None)
    )
    with remote_context as remote_span:
        try:
            yield span
        except Exception as exc:
            if remote_span is not None:
                remote_span.update(level="ERROR", status_message=str(exc))
            raise
        finally:
            span["duration_seconds"] = perf_counter() - started
            if remote_span is not None:
                remote_span.update(output=span, metadata=span["attributes"])
