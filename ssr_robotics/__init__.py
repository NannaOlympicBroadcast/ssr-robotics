"""ssr-robotics — the *environment side* of the SSR ⇄ OpenArm integration.

The SSR agent (the "brain", in ``ssr-agent``) drives the arm purely by
publishing/consuming :mod:`ssr.bus` events. This package runs *next to the
simulator* (NVIDIA Isaac Lab): it listens for ``arm.action.execute`` events,
steps the real environment, and publishes ``arm.grasp.completed`` /
``arm.action.completed`` events (with a TiledCamera frame and grasp metrics)
that wake the agent's bus-handler turns.

Modules
-------
* :mod:`ssr_robotics.protocol`   — shared bus topics + action helpers (from ssr).
* :mod:`ssr_robotics.isaac_env`  — adapter for ``Isaac-Lift-*-OpenArm`` envs.
* :mod:`ssr_robotics.env_runner` — bridge an env to a (remote or in-proc) bus.
* :mod:`ssr_robotics.run_bridge` — CLI entry point (``ssr-arm-bridge``).

``isaac_env`` and ``run_bridge`` import ``isaaclab`` / ``gymnasium`` / ``torch``
lazily, so importing this package does not require a GPU.
"""

from __future__ import annotations

from . import protocol
from .env_runner import EnvRunner

__all__ = ["protocol", "EnvRunner"]
