"""Factory for the ``ask_user`` stock tool.

Call :func:`make_question_tool` once per agent instance during setup.
The returned callable is decorated with ``@tool`` and ready to pass to
``ToolRegistry.register``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable

from pydantic import BaseModel, model_validator

from fipsagents.baseagent.events import QuestionAsked
from fipsagents.baseagent.tools import tool

logger = logging.getLogger("fipsagents.question_tool")


class QuestionOption(BaseModel):
    """A single option presented in a question."""

    label: str
    description: str | None = None
    value: str | None = None

    @model_validator(mode="after")
    def _default_value_to_label(self) -> "QuestionOption":
        """Set value to label if not explicitly provided."""
        if self.value is None:
            self.value = self.label
        return self


class QuestionAnswer(BaseModel):
    """Answer provided by the operator."""

    selected: list[str]
    custom_text: str | None = None


def _generate_question_id() -> str:
    """Sortable timestamp+random ID for question references."""
    ms = int(time.time() * 1000)
    rand = os.urandom(6).hex()
    return f"q_{ms:012x}_{rand}"


def _emit_question(agent: object, event: object) -> None:
    """Append *event* to ``agent._question_events`` defensively.

    If the attribute is absent (e.g. in unit tests that stub only part of
    the contract), the emit is a no-op rather than a crash.
    """
    buf = getattr(agent, "_question_events", None)
    if buf is not None:
        buf.append(event)


def make_question_tool(agent: object) -> Callable:
    """Build the per-agent ``ask_user`` tool function.

    The returned callable is ``@tool``-decorated and ready for
    ``ToolRegistry.register``.
    """

    @tool(
        description=(
            "Ask the operator a structured question. The agent loop pauses "
            "until the operator responds. Use this when you need "
            "clarification, confirmation, or a choice before proceeding."
        ),
        visibility="llm_only",
        name="ask_user",
    )
    async def ask_user(
        prompt: str,
        options: list[dict[str, Any]],
        multiple: bool = False,
        allow_custom: bool = False,
    ) -> str:
        """Ask the user a structured question and wait for their response.

        Args:
            prompt: The question to ask.
            options: List of option dicts. Each must have 'label' (str),
                optional 'description' (str), optional 'value' (str).
                If 'value' is omitted, it defaults to 'label'.
            multiple: Whether the user may select multiple options.
            allow_custom: Whether the user may provide custom text in addition
                to or instead of selecting an option.

        Returns:
            JSON sentinel with ``__pending__: true`` and the question details.
            The agent loop will pause and wait for the operator's response.
        """
        parsed = [QuestionOption(**o) for o in options]

        question_id = _generate_question_id()
        serialized_options = [o.model_dump() for o in parsed]

        pending = {
            "question_id": question_id,
            "prompt": prompt,
            "options": serialized_options,
            "multiple": multiple,
            "allow_custom": allow_custom,
        }

        agent._question_pending = pending

        _emit_question(
            agent,
            QuestionAsked(
                question_id=question_id,
                question_text=prompt,
                options=serialized_options,
                multiple=multiple,
                allow_custom=allow_custom,
            ),
        )

        logger.info(
            "Question posed: %s (id=%s, %d options)",
            prompt[:80],
            question_id,
            len(parsed),
        )

        return json.dumps(
            {
                "__pending__": True,
                "question_id": question_id,
                "prompt": prompt,
                "options": serialized_options,
                "multiple": multiple,
                "allow_custom": allow_custom,
            }
        )

    return ask_user
