"""BaseAgent framework for building production-ready AI agents."""

__version__ = "0.1.0"

from base_agent.agent import BaseAgent, StepOutcome, StepResult
from base_agent.config import AgentConfig, ConfigError, load_config, load_config_from_string
from base_agent.llm import LLMClient, LLMError, ModelResponse
from base_agent.memory import create_memory_client
from base_agent.prompts import Prompt, PromptLoader
from base_agent.rules import Rule, RuleLoader
from base_agent.skills import Skill, SkillLoader
from base_agent.tools import ToolCall, ToolRegistry, ToolResult, tool

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
