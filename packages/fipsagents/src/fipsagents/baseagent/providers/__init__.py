"""LLM provider backends for BaseAgent.

Three backends are available:

- ``openai`` — direct ``AsyncOpenAI`` (any OpenAI-compatible endpoint)
- ``litellm`` — LiteLLM proxy (100+ providers via one dependency)
- ``anthropic`` — direct ``AsyncAnthropic`` (native Anthropic features)

Use :func:`create_provider` to instantiate the right backend from config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fipsagents.baseagent.providers._base import LLMProvider

if TYPE_CHECKING:
    from fipsagents.baseagent.config import LLMConfig

__all__ = ["LLMProvider", "create_provider"]


def create_provider(config: LLMConfig) -> LLMProvider:
    """Instantiate the provider backend for the given config.

    The ``bedrock`` and ``azure`` provider values are deprecated — they
    route through the OpenAI provider pointed at the adapter sidecar.
    Use ``provider: litellm`` with LiteLLM's model-name prefixes instead.

    Raises
    ------
    LLMError
        If the provider's SDK is not installed.
    ValueError
        If ``config.provider`` is not recognized.
    """
    import warnings

    from fipsagents.baseagent.llm import LLMError

    provider = config.provider

    if provider in ("bedrock", "azure"):
        warnings.warn(
            f'provider="{provider}" is deprecated and routes through '
            f"the adapter sidecar. Use provider=\"litellm\" with "
            f"LiteLLM's model-name prefixes (e.g. "
            f'"bedrock/anthropic.claude-3-sonnet") instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        from fipsagents.baseagent.providers._openai import OpenAIProvider
        return OpenAIProvider(config)

    if provider == "openai":
        from fipsagents.baseagent.providers._openai import OpenAIProvider
        return OpenAIProvider(config)

    if provider == "litellm":
        try:
            from fipsagents.baseagent.providers._litellm import LiteLLMProvider
        except ImportError:
            raise LLMError(
                'provider="litellm" requires the litellm SDK. '
                "Install it: pip install fipsagents[litellm]"
            )
        return LiteLLMProvider(config)

    if provider == "anthropic":
        try:
            from fipsagents.baseagent.providers._anthropic import AnthropicProvider
        except ImportError:
            raise LLMError(
                'provider="anthropic" requires the anthropic SDK. '
                "Install it: pip install fipsagents[anthropic]"
            )
        return AnthropicProvider(config)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        f"Valid values: litellm, openai, anthropic"
    )
