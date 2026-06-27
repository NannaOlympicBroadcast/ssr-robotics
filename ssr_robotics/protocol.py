"""Shared bus protocol for arm control.

Re-exports the canonical definitions from ``ssr-agent``
(:mod:`ssr.robotics.protocol`) so the brain and the robot never drift. The
``ssr-agent`` package is a required peer dependency of this bridge.
"""

from __future__ import annotations

from ssr.robotics.protocol import (  # noqa: F401
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    PATTERN_COMPLETED,
    TOPIC_ACTION_COMPLETED,
    TOPIC_ACTION_EXECUTE,
    TOPIC_CAPS,
    TOPIC_CAPS_REQUEST,
    TOPIC_GRASP_COMPLETED,
    TOPIC_RESET,
    TOPIC_STATE,
    TOPIC_STATE_REQUEST,
    TOPIC_TASK_SUCCESS,
    ArmActionRequest,
)

__all__ = [
    "GRIPPER_OPEN", "GRIPPER_CLOSE", "PATTERN_COMPLETED",
    "TOPIC_ACTION_EXECUTE", "TOPIC_ACTION_COMPLETED", "TOPIC_GRASP_COMPLETED",
    "TOPIC_TASK_SUCCESS", "TOPIC_RESET", "TOPIC_STATE", "TOPIC_STATE_REQUEST",
    "TOPIC_CAPS", "TOPIC_CAPS_REQUEST", "ArmActionRequest",
]
