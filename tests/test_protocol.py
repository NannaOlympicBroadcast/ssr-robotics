"""Sanity tests for the env-side bridge that don't need a GPU / Isaac Sim.

They verify the shared protocol is wired through from ``ssr-agent`` and that the
:class:`~ssr_robotics.env_runner.EnvRunner` advertises capabilities and serves a
skill invocation over the real in-process :class:`ssr.bus.core.MessageBus` using
a stub env. The stub implements only the documented env interface — the *real*
env is :class:`~ssr_robotics.isaac_env.IsaacOpenArmEnv`, tested on the robot.
"""

from __future__ import annotations

import base64
import threading

from ssr.bus.core import MessageBus

from ssr_robotics import protocol as P
from ssr_robotics.env_runner import EnvRunner


def test_protocol_reexported_from_ssr():
    assert P.TOPIC_GRASP_COMPLETED == "arm.grasp.completed"
    assert P.TOPIC_CAPS == "arm.capabilities"
    req = P.ArmActionRequest(seq_id="s", episode=1, command="pick", args={"object": "apple"})
    assert P.ArmActionRequest.from_payload(req.to_payload()).command == "pick"


class _StubEnv:
    """Implements the exact env interface EnvRunner depends on (not a sim)."""

    CAM_W, CAM_H = 8, 4

    def __init__(self):
        self.holding = None

    def reset(self):
        self.holding = None

    def capabilities(self):
        return {"action_space": {"dof": 8, "type": "ik+gripper"},
                "skills": [{"name": "pick", "args": {"object": "str"}}],
                "objects": {"apple": [0.5, -0.1, 0.055]}, "camera": {"width": 8, "height": 4}}

    def execute(self, req):
        if req.command == "pick":
            self.holding = req.args.get("object")
        return {"command": req.command, "ok": True, "holding": self.holding,
                "objects": {"apple": [0.5, -0.1, 0.3]},
                "grasp": {"grasped": self.holding is not None}}

    def metrics(self):
        return {"objects": {"apple": [0.5, -0.1, 0.055]}, "holding": self.holding,
                "grasped": False, "gripper_width": 0.044, "object_height": 0.055}

    def frame(self):
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def test_runner_advertises_capabilities():
    bus = MessageBus(name="t", source="t")
    got = {}
    done = threading.Event()
    bus.subscribe(P.TOPIC_CAPS, lambda ev: (got.update(ev.payload), done.set()))
    runner = EnvRunner(bus, _StubEnv()).start()  # advertises on start
    assert done.wait(2.0)
    runner.stop()
    assert got["action_space"]["dof"] == 8
    assert got["skills"][0]["name"] == "pick"


def test_runner_serves_skill_and_reports_completion():
    bus = MessageBus(name="t", source="t")
    runner = EnvRunner(bus, _StubEnv()).start()
    done = threading.Event()
    out = {}
    bus.subscribe(P.TOPIC_GRASP_COMPLETED, lambda ev: (out.update(p=ev.payload), done.set()))
    req = P.ArmActionRequest(seq_id="", episode=1, command="pick", args={"object": "apple"})
    bus.publish(P.TOPIC_ACTION_EXECUTE, req.to_payload())
    assert done.wait(3.0)
    runner.stop()
    assert out["p"]["ok"] is True and out["p"]["holding"] == "apple"
    assert out["p"]["camera"] == {"width": 8, "height": 4}
    assert base64.b64decode(out["p"]["frame_b64"]).startswith(b"\x89PNG")
