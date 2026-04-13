"""BaseAgent — the core integration layer for production-ready AI agents.

Wires together LLM communication, tool dispatch, prompt/skill/rule loading,
memory integration, and MCP server connections.  Subclasses implement
``step()`` with ~20-30 lines of agent logic; everything else is here.

Lifecycle: ``setup()`` -> ``run()`` (loops ``step()``) -> ``shutdown()``
"""

from __future__ import annotations

import abc
import asyncio
import enum
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, TypeVar

from base_agent.config import AgentConfig, load_config
from base_agent.llm import LLMClient, ModelResponse
from base_agent.memory import MemoryClientBase, NullMemoryClient, create_memory_client
from base_agent.prompts import PromptLoader, PromptNotFoundError
from base_agent.rules import RuleLoader
from base_agent.skills import SkillLoader
from base_agent.tools import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Step result — returned by each step() invocation
# ---------------------------------------------------------------------------


class StepOutcome(enum.Enum):
    """Whether the agent loop should continue or stop."""

    CONTINUE = "continue"
    DONE = "done"


@dataclass
class StepResult:
    """Outcome of a single agent step."""

    outcome: StepOutcome
    result: Any = None

    @classmethod
    def continue_(cls) -> StepResult:
        return cls(outcome=StepOutcome.CONTINUE)

    @classmethod
    def done(cls, result: Any = None) -> StepResult:
        return cls(outcome=StepOutcome.DONE, result=result)


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------


class BaseAgent(abc.ABC):
    """Abstract base for all agents.

    Subclasses implement :meth:`step` — one iteration of agent logic.
    Everything else (LLM, tools, prompts, MCP, memory, lifecycle) is
    provided here.
    """

    def __init__(
        self,
        config_path: str | Path = "agent.yaml",
        *,
        config: AgentConfig | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        self._config_path = Path(config_path)
        self._provided_config = config
        self._base_dir = Path(base_dir) if base_dir else None

        # Subsystem instances — populated by setup().
        self.config: AgentConfig | None = None
        self.llm: LLMClient | None = None
        self.tools: ToolRegistry = ToolRegistry()
        self.prompts: PromptLoader = PromptLoader()
        self.skills: SkillLoader = SkillLoader()
        self.rules: RuleLoader = RuleLoader()
        self.memory: MemoryClientBase = NullMemoryClient()

        # Conversation state.
        self.messages: list[dict[str, Any]] = []

        # MCP client references for cleanup.
        self._mcp_clients: list[Any] = []

        # Tracks whether setup has completed.
        self._setup_done = False

    # -- Lifecycle -----------------------------------------------------------

    async def setup(self) -> None:
        """Initialise all subsystems.  Call once before :meth:`run`."""
        # 1. Configuration
        if self._provided_config is not None:
            self.config = self._provided_config
        else:
            self.config = load_config(self._config_path)

        base = self._base_dir or self._config_path.parent

        # 2. Logging
        logging.basicConfig(level=self.config.logging.level)

        logger.info(
            "Setting up agent — model=%s, endpoint=%s",
            self.config.model.name,
            self.config.model.endpoint,
        )

        # 3. LLM client
        self.llm = LLMClient(self.config.model)

        # 4. Tool discovery
        tools_dir = base / self.config.tools.local_dir
        discovered = self.tools.discover(tools_dir)
        logger.info("Discovered %d local tool(s)", len(discovered))

        # 5. Prompts
        prompts_dir = base / self.config.prompts.dir
        if prompts_dir.is_dir():
            loaded = self.prompts.load_all(prompts_dir)
            logger.info("Loaded %d prompt(s)", len(loaded))
        else:
            logger.debug("Prompts directory does not exist: %s", prompts_dir)

        # 6. Skills
        skills_dir = base / "skills"
        if skills_dir.is_dir():
            stubs = self.skills.load_all(skills_dir)
            logger.info("Discovered %d skill stub(s)", len(stubs))
        else:
            logger.debug("Skills directory does not exist: %s", skills_dir)

        # 7. Rules
        rules_dir = base / "rules"
        if rules_dir.is_dir():
            loaded_rules = self.rules.load_all(rules_dir)
            logger.info("Loaded %d rule(s)", len(loaded_rules))
        else:
            logger.debug("Rules directory does not exist: %s", rules_dir)

        # 8. Memory
        memory_cfg_path = base / self.config.memory.config_path
        self.memory = await create_memory_client(memory_cfg_path)

        # 9. MCP servers
        for mcp_cfg in self.config.mcp_servers:
            await self.connect_mcp(mcp_cfg.url)

        self._setup_done = True
        logger.info("Agent setup complete")

    async def run(self) -> Any:
        """Execute the agent loop until DONE or max iterations."""
        if not self._setup_done:
            raise RuntimeError(
                "Agent.run() called before setup(). Call setup() first, "
                "or use start() for the full lifecycle."
            )

        max_iter = self.config.loop.max_iterations
        backoff_cfg = self.config.loop.backoff
        consecutive_errors = 0

        for iteration in range(1, max_iter + 1):
            logger.debug("Step %d/%d", iteration, max_iter)

            try:
                result = await self.step()
            except Exception:
                consecutive_errors += 1
                delay = min(
                    backoff_cfg.initial * (backoff_cfg.multiplier ** (consecutive_errors - 1)),
                    backoff_cfg.max,
                )
                logger.exception(
                    "Step %d raised an exception — backing off %.1fs "
                    "(consecutive errors: %d)",
                    iteration,
                    delay,
                    consecutive_errors,
                )
                await asyncio.sleep(delay)
                continue

            # Reset error counter on a successful step.
            consecutive_errors = 0

            if result.outcome is StepOutcome.DONE:
                logger.info(
                    "Agent completed after %d step(s)", iteration
                )
                return result.result

        logger.warning(
            "Agent hit max iterations (%d) without completing", max_iter
        )
        return None

    async def shutdown(self) -> None:
        """Clean up resources: close MCP connections and any open handles."""
        logger.info("Shutting down agent")
        for client in self._mcp_clients:
            try:
                if hasattr(client, "close"):
                    await client.close()
                elif hasattr(client, "disconnect"):
                    await client.disconnect()
            except Exception:
                logger.warning(
                    "Error closing MCP client", exc_info=True
                )
        self._mcp_clients.clear()
        self._setup_done = False
        logger.info("Agent shutdown complete")

    async def start(self) -> Any:
        """Full lifecycle: setup -> run -> shutdown (with guaranteed cleanup)."""
        try:
            await self.setup()
            return await self.run()
        finally:
            await self.shutdown()

    # -- Abstract method -----------------------------------------------------

    @abc.abstractmethod
    async def step(self) -> StepResult:
        """One iteration of agent logic (subclasses implement this)."""
        ...

    # -- Conversation state --------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict[str, Any]]:
        """Return a copy of the current conversation history."""
        return list(self.messages)

    def clear_messages(self) -> None:
        """Reset the conversation history."""
        self.messages.clear()

    # -- LLM convenience methods ---------------------------------------------
    # These delegate to self.llm but automatically include conversation state
    # and tool schemas when appropriate.

    async def call_model(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        tools: list[dict[str, Any]] | None = None,
        include_tools: bool = True,
        **kwargs: Any,
    ) -> ModelResponse:
        """Chat completion.  Defaults to ``self.messages`` and auto-includes
        LLM-visible tool schemas unless *include_tools* is ``False``."""
        self._require_llm()
        msgs = messages if messages is not None else self.messages
        if include_tools and tools is None:
            schemas = self.get_tool_schemas()
            tools = schemas if schemas else None
        return await self.llm.call_model(msgs, tools=tools, **kwargs)

    async def call_model_json(
        self,
        schema: Any,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Structured-output completion.  Returns parsed/validated object."""
        self._require_llm()
        msgs = messages if messages is not None else self.messages
        return await self.llm.call_model_json(msgs, schema, **kwargs)

    async def call_model_stream(
        self,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Streaming completion.  Yields content chunks."""
        self._require_llm()
        msgs = messages if messages is not None else self.messages
        async for chunk in self.llm.call_model_stream(msgs, **kwargs):
            yield chunk

    async def call_model_validated(
        self,
        validator_fn: Callable[[ModelResponse], T],
        messages: list[dict[str, Any]] | None = None,
        *,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> T:
        """Call model, validate response, retry with backoff on failure."""
        self._require_llm()
        msgs = messages if messages is not None else self.messages
        return await self.llm.call_model_validated(
            msgs, validator_fn, max_retries=max_retries, **kwargs
        )

    def _require_llm(self) -> None:
        """Guard against calling LLM methods before setup."""
        if self.llm is None:
            raise RuntimeError(
                "LLM client not initialised. Call setup() before making "
                "model calls."
            )

    # -- Tool dispatch -------------------------------------------------------

    async def use_tool(self, name: str, **kwargs: Any) -> ToolResult:
        """Call a tool through the registry.

        This is the single dispatch point for all agent-code tool calls
        (plane 1).  Logging is applied around the call.
        """
        logger.info("Tool call: %s(%s)", name, _summarise_kwargs(kwargs))
        result = await self.tools.execute(name, **kwargs)
        if result.is_error:
            logger.warning("Tool %s failed: %s", name, result.error)
        else:
            logger.debug("Tool %s returned: %s", name, _truncate(result.result))
        return result

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schemas for LLM-visible tools."""
        return self.tools.generate_schemas()

    # -- MCP integration -----------------------------------------------------

    async def connect_mcp(self, server_url: str) -> None:
        """Connect to an MCP server via FastMCP v3 and register its tools."""
        logger.info("Connecting to MCP server: %s", server_url)
        try:
            from fastmcp import Client as McpClient

            client = McpClient(server_url)
            await client.__aenter__()

            # Discover tools from the server.
            tools_list = await client.list_tools()
            registered = 0
            for mcp_tool in tools_list:
                # Wrap MCP tool as a local callable and register it.
                _register_mcp_tool(self.tools, client, mcp_tool)
                registered += 1

            self._mcp_clients.append(client)
            logger.info(
                "Connected to MCP server %s — registered %d tool(s)",
                server_url,
                registered,
            )
        except ImportError:
            logger.warning(
                "fastmcp package not installed — cannot connect to MCP "
                "server %s. Install with: pip install fastmcp",
                server_url,
            )
        except Exception:
            logger.exception(
                "Failed to connect to MCP server: %s", server_url
            )

    # -- System prompt assembly -----------------------------------------------

    def build_system_prompt(self) -> str:
        """Assemble system prompt from main prompt, rules, and skills."""
        sections: list[str] = []

        # 1. Main system prompt.
        try:
            system_prompt = self.prompts.get("system")
            sections.append(system_prompt.render())
        except PromptNotFoundError:
            logger.debug("No 'system' prompt found — skipping")

        # 2. Rules.
        rules_text = self.rules.get_combined_content()
        if rules_text:
            sections.append(rules_text)

        # 3. Activated skill manifests.
        manifest = self.skills.get_manifest()
        if manifest:
            skill_lines = ["# Available Skills", ""]
            for entry in manifest:
                triggers = ", ".join(entry.triggers) if entry.triggers else "none"
                skill_lines.append(
                    f"- **{entry.name}**: {entry.description} "
                    f"(triggers: {triggers})"
                )
            sections.append("\n".join(skill_lines))

        return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# MCP tool registration helper
# ---------------------------------------------------------------------------


def _register_mcp_tool(
    registry: ToolRegistry, client: Any, mcp_tool: Any,
) -> None:
    """Wrap an MCP tool as a local callable and register it (llm_only)."""
    from base_agent.tools import ToolMeta, _TOOL_MARKER

    tool_name = mcp_tool.name
    tool_desc = getattr(mcp_tool, "description", "") or tool_name
    input_schema = getattr(mcp_tool, "inputSchema", None) or {}

    async def _call_mcp_tool(**kwargs: Any) -> str:
        result = await client.call_tool(tool_name, kwargs)
        return str(result)

    meta = ToolMeta(
        name=tool_name,
        description=tool_desc,
        visibility="llm_only",
        fn=_call_mcp_tool,
        is_async=True,
        parameters=input_schema,
    )
    setattr(_call_mcp_tool, _TOOL_MARKER, meta)

    try:
        registry.register(_call_mcp_tool)
    except ValueError:
        logger.warning(
            "MCP tool %r conflicts with an existing tool name — skipping",
            tool_name,
        )


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------


def _summarise_kwargs(kwargs: dict[str, Any], max_len: int = 120) -> str:
    """Produce a compact string summary of kwargs for log messages."""
    if not kwargs:
        return ""
    parts = [f"{k}={_truncate(repr(v), 40)}" for k, v in kwargs.items()]
    joined = ", ".join(parts)
    return _truncate(joined, max_len)


def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate a string and append '...' if it exceeds *max_len*."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
