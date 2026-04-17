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

from fipsagents.baseagent.config import AgentConfig, McpServerConfig, load_config
from fipsagents.baseagent.events import (
    ContentDelta,
    ReasoningDelta,
    StreamComplete,
    StreamEvent,
    StreamMetrics,
    ToolCallDelta,
    ToolResultEvent,
)
from fipsagents.baseagent.llm import LLMClient, ModelResponse
from fipsagents.baseagent.reasoning import ThinkTagParser, create_reasoning_parser
from fipsagents.baseagent.memory import MemoryClientBase, NullMemoryClient, create_memory_client
from fipsagents.baseagent.prompts import PromptLoader, PromptNotFoundError
from fipsagents.baseagent.rules import RuleLoader
from fipsagents.baseagent.skills import SkillLoader
from fipsagents.baseagent.tools import ToolRegistry, ToolResult

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

        # MCP prompts, resources, and resource templates — populated by connect_mcp().
        self._mcp_prompts: dict[str, tuple[Any, Any]] = {}      # name → (client, mcp.types.Prompt)
        self._mcp_resources: dict[str, tuple[Any, Any]] = {}     # uri_string → (client, mcp.types.Resource)
        self._mcp_resource_templates: dict[str, tuple[Any, Any]] = {}  # uri_template → (client, mcp.types.ResourceTemplate)

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

        # 4b. Tool inspection
        if self.config.security.tool_inspection.enabled:
            from fipsagents.baseagent.tool_inspector import ToolInspector

            inspector = ToolInspector()
            effective_mode = (
                self.config.security.tool_inspection.mode
                or self.config.security.mode
            )
            self.tools.set_inspector(inspector, mode=effective_mode)
            logger.info(
                "Tool inspection enabled (mode=%s)", effective_mode
            )

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
        self.memory = await create_memory_client(
            memory_cfg_path, config=self.config.memory
        )

        # 9. MCP servers
        for mcp_cfg in self.config.mcp_servers:
            await self.connect_mcp(mcp_cfg)

        # 10. Seed messages with system prompt + optional memory prefix.
        self.messages.append(
            {"role": "system", "content": self.build_system_prompt()}
        )
        prefix = await self.build_memory_prefix()
        if prefix:
            self.messages.append(
                {"role": self.config.memory.prefix_role, "content": prefix}
            )
            logger.info(
                "Memory prefix injected (%d chars, role=%s)",
                len(prefix),
                self.config.memory.prefix_role,
            )

        # 11. Reasoning parser for models that use <think> tags in content.
        self._reasoning_parser: ThinkTagParser | None = create_reasoning_parser(
            self.config.model.name
        )
        if self._reasoning_parser:
            logger.info(
                "Think-tag reasoning parser enabled for model %s",
                self.config.model.name,
            )

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
        self._mcp_prompts.clear()
        self._mcp_resources.clear()
        self._mcp_resource_templates.clear()
        self._setup_done = False
        logger.info("Agent shutdown complete")

    async def start(self) -> Any:
        """Full lifecycle: setup -> run -> shutdown (with guaranteed cleanup)."""
        try:
            await self.setup()
            return await self.run()
        finally:
            await self.shutdown()

    # -- Step: one iteration of agent logic ---------------------------------

    async def step(self) -> StepResult:
        """One iteration of agent logic.

        The default implementation consumes :meth:`astep_stream` and returns
        the concatenated ``ContentDelta`` content as a ``StepResult.done``.
        Subclasses typically override ``astep_stream`` only; both sync and
        streaming clients then share the same ReAct loop, tool dispatch, and
        any pre/post-turn hooks (memory recall, system prompt injection).

        Override this method directly only when a subclass needs sync-specific
        behavior that doesn't make sense to expose as events — most agents
        should not.
        """
        content_parts: list[str] = []
        async for event in self.astep_stream():
            if isinstance(event, ContentDelta):
                content_parts.append(event.content)
        return StepResult.done("".join(content_parts))

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

    async def astep_stream(
        self,
        *,
        max_iterations: int = 10,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming agent loop. Yields typed ``StreamEvent`` values.

        Drives the model in streaming mode and emits:

        - ``ReasoningDelta`` for each ``delta.reasoning_content`` chunk
          (models like gpt-oss-20b expose this natively)
        - ``ToolCallDelta`` for each incremental tool-call chunk the
          model emits, including ``arguments`` streamed token-by-token
        - ``ToolResultEvent`` after the agent executes each tool
        - ``ContentDelta`` for each ``delta.content`` chunk (the
          user-visible response)
        - ``StreamComplete`` as the terminal event, carrying
          ``StreamMetrics`` (TTFT, ITL samples, totals)

        The loop terminates when the model returns a turn with
        ``finish_reason`` other than ``"tool_calls"``. Subclasses that
        want custom pre/post-turn work (memory recall, message
        injection) should override this method and call ``super()``.

        This is source-agnostic: tools from MCP servers and local
        ``@tool`` functions flow through the same dispatch point, so
        streaming looks identical regardless of tool origin.
        """
        self._require_llm()

        metrics = StreamMetrics()
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        last_content_time: float | None = None

        def _mark_first_reasoning() -> None:
            if metrics.time_to_first_reasoning is None:
                metrics.time_to_first_reasoning = loop.time() - start_time

        def _mark_content(now: float) -> None:
            nonlocal last_content_time
            if metrics.time_to_first_content is None:
                metrics.time_to_first_content = now - start_time
            if last_content_time is not None:
                metrics.inter_token_latencies.append(now - last_content_time)
            last_content_time = now

        finish_reason = "stop"

        for _ in range(max_iterations):
            metrics.model_calls += 1
            schemas = self.get_tool_schemas()
            tools_arg = schemas if schemas else None

            # Accumulators for this turn. Keyed by tool_call index since
            # OpenAI streams multiple concurrent tool calls interleaved.
            tool_buf: dict[int, dict[str, Any]] = {}
            assistant_content_parts: list[str] = []
            if self._reasoning_parser:
                self._reasoning_parser.reset()

            async for chunk in self.llm.call_model_stream_raw(
                self.messages, tools=tools_arg
            ):
                try:
                    choice = chunk.choices[0]
                except (AttributeError, IndexError):
                    continue
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                # Reasoning ("thinking") deltas.
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    _mark_first_reasoning()
                    yield ReasoningDelta(content=reasoning)

                # Content deltas. If a reasoning parser is active,
                # separate <think>…</think> blocks from visible content.
                content = getattr(delta, "content", None)
                if content:
                    if self._reasoning_parser:
                        for kind, text in self._reasoning_parser.feed(content):
                            if kind == "reasoning":
                                _mark_first_reasoning()
                                yield ReasoningDelta(content=text)
                            else:
                                now = loop.time()
                                _mark_content(now)
                                assistant_content_parts.append(text)
                                yield ContentDelta(content=text)
                    else:
                        now = loop.time()
                        _mark_content(now)
                        assistant_content_parts.append(content)
                        yield ContentDelta(content=content)

                # Tool-call deltas. OpenAI streams these with an
                # ``index`` so concurrent calls stay distinct.
                tc_list = getattr(delta, "tool_calls", None) or []
                for tc in tc_list:
                    idx = getattr(tc, "index", 0) or 0
                    buf = tool_buf.setdefault(
                        idx, {"id": None, "name": None, "arguments": ""}
                    )
                    tc_id = getattr(tc, "id", None)
                    fn = getattr(tc, "function", None)
                    tc_name = getattr(fn, "name", None) if fn else None
                    tc_args = getattr(fn, "arguments", None) if fn else None

                    # First delta for this index usually carries id+name.
                    first = buf["id"] is None and tc_id is not None
                    if tc_id and not buf["id"]:
                        buf["id"] = tc_id
                    if tc_name and not buf["name"]:
                        buf["name"] = tc_name
                    if tc_args:
                        buf["arguments"] += tc_args

                    yield ToolCallDelta(
                        index=idx,
                        call_id=buf["id"] if first else None,
                        name=buf["name"] if first else None,
                        arguments_delta=tc_args or "",
                    )

                turn_finish = getattr(choice, "finish_reason", None)
                if turn_finish:
                    finish_reason = turn_finish

            # Flush any buffered reasoning parser state.
            if self._reasoning_parser:
                for kind, text in self._reasoning_parser.flush():
                    if kind == "reasoning":
                        yield ReasoningDelta(content=text)
                    else:
                        assistant_content_parts.append(text)
                        yield ContentDelta(content=text)

            # Extract any usage stats the provider reported. Not all
            # providers send these with streaming.
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                metrics.prompt_tokens = (
                    getattr(usage, "prompt_tokens", None)
                    or metrics.prompt_tokens
                )
                metrics.completion_tokens = (
                    getattr(usage, "completion_tokens", None)
                    or metrics.completion_tokens
                )
                metrics.total_tokens = (
                    getattr(usage, "total_tokens", None)
                    or metrics.total_tokens
                )

            # If the model decided to call tools, execute them and loop.
            if tool_buf:
                import json as _json

                assembled_calls = []
                for idx in sorted(tool_buf.keys()):
                    buf = tool_buf[idx]
                    if not buf["id"] or not buf["name"]:
                        continue
                    assembled_calls.append(
                        {
                            "id": buf["id"],
                            "type": "function",
                            "function": {
                                "name": buf["name"],
                                "arguments": buf["arguments"],
                            },
                        }
                    )

                # Append the assistant's tool-calling message so the
                # conversation history is correctly shaped for the next
                # model call.
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": "".join(assistant_content_parts) or "",
                        "tool_calls": assembled_calls,
                    }
                )

                # Execute each tool and emit the result event.
                for call in assembled_calls:
                    metrics.tool_calls += 1
                    fn_name = call["function"]["name"]
                    try:
                        args = (
                            _json.loads(call["function"]["arguments"])
                            if call["function"]["arguments"]
                            else {}
                        )
                    except _json.JSONDecodeError:
                        args = {}

                    result = await self.tools.execute(fn_name, **args)
                    is_err = result.is_error
                    content_str = (
                        result.result
                        if not is_err
                        else f"ERROR: {result.error}"
                    )

                    self.messages.append(
                        {
                            "role": "tool",
                            "content": content_str,
                            "tool_call_id": call["id"],
                        }
                    )
                    yield ToolResultEvent(
                        call_id=call["id"],
                        name=fn_name,
                        content=content_str,
                        is_error=is_err,
                    )

                # Continue the loop: call the model again with the tool
                # results appended.
                continue

            # No tool calls -> this turn produced the final response.
            if assistant_content_parts:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": "".join(assistant_content_parts),
                    }
                )
            break
        else:
            # Loop exhausted without break -> hit iteration cap.
            finish_reason = "length"

        metrics.total_time = loop.time() - start_time
        yield StreamComplete(finish_reason=finish_reason, metrics=metrics)

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

    async def connect_mcp(
        self, target: Any,
    ) -> None:
        """Connect to an MCP server via FastMCP v3 and register its tools, prompts, and resources.

        Parameters
        ----------
        target:
            One of:

            - **str** — URL for HTTP transport (backward-compatible).
            - **McpServerConfig** — HTTP (``url``) or stdio (``command``).
            - **FastMCP** — in-process server object (no subprocess or
              network; FastMCP v3 ``FastMCPTransport``).
        """
        # Resolve the transport argument for the FastMCP Client.
        if isinstance(target, str):
            label = target
            transport: Any = target
        elif isinstance(target, McpServerConfig):
            if target.url:
                label = target.url
                transport = target.url
            else:
                label = f"stdio:{target.command}"
                from fastmcp.client.transports import StdioTransport

                transport = StdioTransport(
                    command=target.command,
                    args=target.args,
                    env=target.env,
                    cwd=target.cwd,
                )
        else:
            # Assume it's a FastMCP server instance (or any object that
            # FastMCP Client can auto-detect as a transport).
            label = getattr(target, "name", None) or type(target).__name__
            transport = target

        logger.info("Connecting to MCP server: %s", label)
        try:
            from fastmcp import Client as McpClient

            client = McpClient(transport)
            await client.__aenter__()

            # Discover tools from the server.
            tools_list = await client.list_tools()
            registered = 0
            for mcp_tool in tools_list:
                # Wrap MCP tool as a local callable and register it.
                _register_mcp_tool(self.tools, client, mcp_tool)
                registered += 1

            # Discover prompts.
            prompt_count = 0
            try:
                prompts_list = await client.list_prompts()
                for mcp_prompt in prompts_list:
                    pname = mcp_prompt.name
                    if pname in self._mcp_prompts:
                        logger.warning(
                            "MCP prompt %r already registered — skipping duplicate from %s",
                            pname, label,
                        )
                        continue
                    self._mcp_prompts[pname] = (client, mcp_prompt)
                    prompt_count += 1
            except Exception:
                logger.debug("MCP server %s does not expose prompts (or error listing them)", label, exc_info=True)

            # Discover resources.
            resource_count = 0
            try:
                resources_list = await client.list_resources()
                for mcp_resource in resources_list:
                    uri_str = str(mcp_resource.uri)
                    if uri_str in self._mcp_resources:
                        logger.warning(
                            "MCP resource %r already registered — skipping duplicate from %s",
                            uri_str, label,
                        )
                        continue
                    self._mcp_resources[uri_str] = (client, mcp_resource)
                    resource_count += 1
            except Exception:
                logger.debug("MCP server %s does not expose resources (or error listing them)", label, exc_info=True)

            # Discover resource templates.
            template_count = 0
            try:
                templates_list = await client.list_resource_templates()
                for mcp_template in templates_list:
                    tpl_str = mcp_template.uriTemplate
                    if tpl_str in self._mcp_resource_templates:
                        logger.warning(
                            "MCP resource template %r already registered — skipping duplicate from %s",
                            tpl_str, label,
                        )
                        continue
                    self._mcp_resource_templates[tpl_str] = (client, mcp_template)
                    template_count += 1
            except Exception:
                logger.debug("MCP server %s does not expose resource templates (or error listing them)", label, exc_info=True)

            self._mcp_clients.append(client)
            logger.info(
                "Connected to MCP server %s — %d tool(s), %d prompt(s), %d resource(s), %d template(s)",
                label, registered, prompt_count, resource_count, template_count,
            )
        except ImportError:
            logger.warning(
                "fastmcp package not installed — cannot connect to MCP "
                "server %s. Install with: pip install fastmcp",
                label,
            )
        except Exception:
            logger.exception(
                "Failed to connect to MCP server: %s", label
            )

    async def get_mcp_prompt(
        self, name: str, arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Render an MCP prompt by name.

        Calls the originating MCP server's ``get_prompt`` method.

        Returns
        -------
        mcp.types.GetPromptResult
            Contains ``messages`` (list of PromptMessage) and optional
            ``description``.

        Raises
        ------
        KeyError
            If *name* is not a discovered MCP prompt.
        """
        if name not in self._mcp_prompts:
            raise KeyError(
                f"MCP prompt {name!r} not found. "
                f"Available: {sorted(self._mcp_prompts)}"
            )
        client, _prompt_meta = self._mcp_prompts[name]
        return await client.get_prompt(name, arguments=arguments)

    async def read_resource(self, uri: str) -> Any:
        """Read an MCP resource by URI.

        Calls the originating MCP server's ``read_resource`` method.

        Returns
        -------
        list[mcp.types.TextResourceContents | mcp.types.BlobResourceContents]

        Raises
        ------
        KeyError
            If *uri* is not a discovered MCP resource.
        """
        if uri not in self._mcp_resources:
            raise KeyError(
                f"MCP resource {uri!r} not found. "
                f"Available: {sorted(self._mcp_resources)}"
            )
        client, _resource_meta = self._mcp_resources[uri]
        return await client.read_resource(uri)

    def list_mcp_prompts(self) -> list[dict[str, Any]]:
        """Return metadata for all discovered MCP prompts."""
        result = []
        for name, (_client, prompt) in sorted(self._mcp_prompts.items()):
            args = getattr(prompt, "arguments", None) or []
            entry: dict[str, Any] = {
                "name": prompt.name,
                "description": getattr(prompt, "description", None) or "",
                "arguments": [
                    {
                        "name": a.name,
                        "description": getattr(a, "description", None) or "",
                        "required": getattr(a, "required", None),
                    }
                    for a in args
                ],
            }
            result.append(entry)
        return result

    def list_mcp_resources(self) -> list[dict[str, Any]]:
        """Return metadata for all discovered MCP resources."""
        result = []
        for uri, (_client, resource) in sorted(self._mcp_resources.items()):
            result.append({
                "uri": str(resource.uri),
                "name": resource.name,
                "description": getattr(resource, "description", None) or "",
                "mimeType": getattr(resource, "mimeType", None),
            })
        return result

    def list_mcp_resource_templates(self) -> list[dict[str, Any]]:
        """Return metadata for all discovered MCP resource templates."""
        result = []
        for tpl, (_client, template) in sorted(self._mcp_resource_templates.items()):
            result.append({
                "uriTemplate": template.uriTemplate,
                "name": template.name,
                "description": getattr(template, "description", None) or "",
                "mimeType": getattr(template, "mimeType", None),
            })
        return result

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

    async def build_memory_prefix(self) -> str | None:
        """Return a stable memory block to inject after the system prompt.

        Called once during :meth:`setup`.  The result is inserted as a
        message (role controlled by ``config.memory.prefix_role``) at
        index 1 in ``self.messages``, immediately after the system prompt.
        It stays pinned there for the life of the session — never
        re-queried per turn — so inference-server prefix caches stay warm.

        The default implementation calls ``self.memory.search("")`` to
        retrieve all memories in backend-native order, joins their
        ``content`` fields, and truncates at
        ``config.memory.max_prefix_chars``.  Returns ``None`` when the
        backend produces no results (including ``NullMemoryClient``).

        Subclasses override this to customise the query, formatting, or
        to return ``None`` unconditionally if they prefer per-turn recall.
        """
        results = await self.memory.search("")
        if not results:
            return None

        parts = [r.get("content", "") for r in results]
        parts = [p for p in parts if p.strip()]  # drop blanks
        if not parts:
            return None

        joined = "\n\n---\n\n".join(parts)

        limit = self.config.memory.max_prefix_chars
        if limit and len(joined) > limit:
            joined = joined[:limit] + "\n\n… [truncated]"

        return joined


# ---------------------------------------------------------------------------
# MCP tool registration helper
# ---------------------------------------------------------------------------


def _register_mcp_tool(
    registry: ToolRegistry, client: Any, mcp_tool: Any,
) -> None:
    """Wrap an MCP tool as a local callable and register it (llm_only)."""
    from fipsagents.baseagent.tools import ToolMeta, _TOOL_MARKER

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
