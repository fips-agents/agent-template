"""BaseAgent framework for building production-ready AI agents."""

__version__ = "0.2.0"

from fipsagents.baseagent.agent import BaseAgent, StepOutcome, StepResult
from fipsagents.baseagent.config import AgentConfig, ConfigError, load_config, load_config_from_string
from fipsagents.baseagent.llm import LLMClient, LLMError, ModelResponse
from fipsagents.baseagent.memory import MemoryClientBase, NullMemoryClient, create_memory_client
from fipsagents.baseagent.prompts import Prompt, PromptLoader
from fipsagents.baseagent.rules import Rule, RuleLoader
from fipsagents.baseagent.skills import Skill, SkillLoader
from fipsagents.baseagent.tools import ToolCall, ToolRegistry, ToolResult, tool

__all__ = [
    # agent
    "BaseAgent",
    "StepOutcome",
    "StepResult",
    # config
    "AgentConfig",
    "ConfigError",
    "load_config",
    "load_config_from_string",
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
    # tools
    "tool",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
]
