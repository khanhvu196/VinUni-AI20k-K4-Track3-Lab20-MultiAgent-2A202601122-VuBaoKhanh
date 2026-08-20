"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Gemini client with centralized retry, timeout, and usage capture."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Validate configuration and return a Gemini completion."""

        if not self.settings.gemini_api_key:
            raise AgentExecutionError("GEMINI_API_KEY is not configured")
        if not system_prompt.strip() or not user_prompt.strip():
            raise AgentExecutionError("System and user prompts must not be empty")

        return self._generate(system_prompt, user_prompt)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    def _generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call Gemini; retry transient provider errors at this boundary."""

        client = genai.Client(
            api_key=self.settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=self.settings.timeout_seconds * 1000),
        )
        response = client.models.generate_content(
            model=self.settings.gemini_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        content = response.text or ""
        if not content.strip():
            raise AgentExecutionError("Gemini returned an empty response")

        usage = response.usage_metadata
        return LLMResponse(
            content=content,
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
            cost_usd=0.0,
        )
