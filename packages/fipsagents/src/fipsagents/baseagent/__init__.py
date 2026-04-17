"""BaseAgent framework for building production-ready AI agents."""

__version__ = "0.5.0.dev0"

from fipsagents.baseagent.agent import BaseAgent, StepOutcome, StepResult
from fipsagents.baseagent.config import AgentConfig, ConfigError, NodeConfig, SecurityConfig, load_config, load_config_from_string
from fipsagents.baseagent.events import (
    ContentDelta,
    ReasoningDelta,
    StreamComplete,
    StreamEvent,
    StreamMetrics,
    ToolCallDelta,
    ToolResultEvent,
)
from fipsagents.baseagent.llm import LLMClient, LLMError, ModelResponse
from fipsagents.baseagent.memory import MemoryClientBase, NullMemoryClient, create_memory_client
from fipsagents.baseagent.prompts import Prompt, PromptLoader
from fipsagents.baseagent.rules import Rule, RuleLoader
from fipsagents.baseagent.skills import Skill, SkillLoader
from fipsagents.baseagent.diagnostics import RoleProbeResult, probe_role_support
from fipsagents.baseagent.tool_inspector import InspectionFinding, InspectionResult, ToolInspector
from fipsagents.baseagent.tools import ToolCall, ToolRegistry, ToolResult, tool

__all__ = [
    # agent
    "BaseAgent",
    "StepOutcome",
    "StepResult",
    # config
    "AgentConfig",
    "ConfigError",
    "NodeConfig",
    "SecurityConfig",
    "load_config",
    "load_config_from_string",
    # events (streaming)
    "ContentDelta",
    "ReasoningDelta",
    "StreamComplete",
    "StreamEvent",
    "StreamMetrics",
    "ToolCallDelta",
    "ToolResultEvent",
    # llm
    "LLMClient",
    "LLMError",
    "ModelResponse",
    # memory
    "MemoryClientBase",
    "NullMemoryClient",
    "create_memory_client",
    # prompts
    "Prompt",
    "PromptLoader",
    # rules
    "Rule",
    "RuleLoader",
    # skills
    "Skill",
    "SkillLoader",
    # diagnostics
    "RoleProbeResult",
    "probe_role_support",
    # tool_inspector
    "InspectionFinding",
    "InspectionResult",
    "ToolInspector",
    # tools
    "tool",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
]
