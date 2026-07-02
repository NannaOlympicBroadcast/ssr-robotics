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
    assert P.TOPIC_GRASP_REQUEST == "arm.grasp.request"
    assert P.TOPIC_GRASP_RESULT == "arm.grasp.result"
    assert P.TOPIC_STREAM_START == "arm.stream.start"
    assert P.TOPIC_CAPS == "arm.capabilities"
    req = P.ArmActionRequest(seq_id="s", episode=1, command="servo", args={"dx": 0.01})
    assert P.ArmActionRequest.from_payload(req.to_payload()).command == "servo"
    greq = P.ArmGraspRequest(seq_id="g", label="apple", target_b64="cGln")
    assert P.ArmGraspRequest.from_payload(greq.to_payload()).label == "apple"


class _StubEnv:
    """Implements the exact env interface EnvRunner depends on (not a sim)."""

    CAM_W, CAM_H = 8, 4

    def __init__(self):
        self.holding = False

    def reset(self):
        self.holding = False

    def capabilities(self):
        return {"action_space": {"dof": 8, "type": "ik+gripper"},
                "skills": [{"name": "servo",
                            "args": {"dx": "float", "dy": "float", "dz": "float",
                                     "grip": "open|close|hold"}}],
                "objects": {"obj0": [0.5, -0.1, 0.055]}, "camera": {"width": 8, "height": 4}}

    def execute(self, req):
        if req.command == "servo" and req.args.get("grip") == "close":
            self.holding = True
        return {"command": req.command, "ok": True, "holding": self.holding,
                "objects": {"obj0": [0.5, -0.1, 0.3]},
                "grasp": {"grasped": self.holding}}

    def metrics(self):
        return {"objects": {"obj0": [0.5, -0.1, 0.055]}, "holding": self.holding,
                "grasped": False, "gripper_width": 0.044, "object_height": 0.055}

    def frame(self):
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

    def frame_rgb(self):
        return b"\x00" * (8 * 4 * 3)


def test_runner_advertises_capabilities():
    bus = MessageBus(name="t", source="t")
    got = {}
    done = threading.Event()
    bus.subscribe(P.TOPIC_CAPS, lambda ev: (got.update(ev.payload), done.set()))
    runner = EnvRunner(bus, _StubEnv()).start()  # advertises on start
    assert done.wait(2.0)
    runner.stop()
    assert got["action_space"]["dof"] == 8
    assert got["skills"][0]["name"] == "servo"


def test_runner_serves_skill_and_reports_completion():
    bus = MessageBus(name="t", source="t")
    runner = EnvRunner(bus, _StubEnv()).start()
    done = threading.Event()
    out = {}
    bus.subscribe(P.TOPIC_ACTION_COMPLETED, lambda ev: (out.update(p=ev.payload), done.set()))
    req = P.ArmActionRequest(seq_id="", episode=1, command="servo",
                             args={"dz": -0.03, "grip": "close"})
    bus.publish(P.TOPIC_ACTION_EXECUTE, req.to_payload())
    # The bus handler only enqueues; pump() (the sim thread's job) runs it.
    assert runner.pump(timeout=2.0)
    assert done.wait(3.0)
    runner.stop()
    assert out["p"]["ok"] is True and out["p"]["holding"] is True
    assert out["p"]["camera"] == {"width": 8, "height": 4}
    assert base64.b64decode(out["p"]["frame_b64"]).startswith(b"\x89PNG")


def test_reset_interrupts_and_drains_queued_jobs():
    """A reset must preempt the backlog: without this, the reset job sits behind a
    long raw replay holding the sim thread and the sim looks frozen until it ends.
    _on_reset interrupts any in-flight skill and drops queued (not-yet-run) jobs."""
    bus = MessageBus(name="t", source="t")
    env = _StubEnv()
    env.interrupted = False
    env.interrupt = lambda: setattr(env, "interrupted", True)
    runner = EnvRunner(bus, env).start()

    # Two execute jobs pile up (handlers only enqueue; pump hasn't run them).
    for _ in range(2):
        req = P.ArmActionRequest(seq_id="", episode=1, command="servo", args={"dx": 0.01})
        bus.publish(P.TOPIC_ACTION_EXECUTE, req.to_payload())

    done = threading.Event()
    bus.subscribe(P.TOPIC_STATE, lambda ev: done.set())
    bus.publish(P.TOPIC_RESET, {"episode": 2})

    assert env.interrupted is True            # in-flight skill asked to stop
    assert runner.pump(timeout=2.0)           # next job out of the queue is the reset
    assert done.wait(2.0)                      # ...which published fresh state
    assert runner.pump(timeout=0.2) is False  # the two execute jobs were dropped
    runner.stop()


def test_env_only_touched_from_pump_thread():
    """The bug this guards against: Isaac Sim's sim/render context is not
    thread-safe and may only be driven from the thread that owns
    ``simulation_app``. Bus callbacks run on whatever thread the transport uses
    (a remote BusClient's own asyncio loop thread, not that thread) -- they must
    only enqueue a job, never call into the env directly. ``pump()`` is the sole
    place env methods may run.
    """
    bus = MessageBus(name="t", source="t")
    env = _StubEnv()
    runner = EnvRunner(bus, env).start()

    publish_thread_id = threading.get_ident()
    call_thread_ids = []
    orig_execute = env.execute

    def _tracking_execute(req):
        call_thread_ids.append(threading.get_ident())
        return orig_execute(req)

    env.execute = _tracking_execute

    req = P.ArmActionRequest(seq_id="", episode=1, command="servo", args={"dx": 0.01})
    bus.publish(P.TOPIC_ACTION_EXECUTE, req.to_payload())
    # publish() dispatches the handler synchronously on this thread, but the
    # handler must only enqueue a job -- the env itself stays untouched so far.
    assert call_thread_ids == []

    pump_thread_id = {}

    def _sim_thread():
        pump_thread_id["id"] = threading.get_ident()
        runner.pump(timeout=3.0)

    t = threading.Thread(target=_sim_thread)
    t.start()
    t.join(5.0)
    runner.stop()

    assert call_thread_ids == [pump_thread_id["id"]]
    assert pump_thread_id["id"] != publish_thread_id


def test_runner_stream_start_without_url_reports_error():
    bus = MessageBus(name="t", source="t")
    runner = EnvRunner(bus, _StubEnv(), stream_url="")
    runner.start()
    done = threading.Event()
    out = {}
    bus.subscribe(P.TOPIC_STREAM_STARTED, lambda ev: (out.update(p=ev.payload), done.set()))
    bus.publish(P.TOPIC_STREAM_START, {"seq_id": "g1"})
    assert runner.pump(timeout=2.0)
    assert done.wait(2.0)
    runner.stop()
    assert out["p"]["ok"] is False and out["p"]["seq_id"] == "g1"
    assert "stream" in out["p"]["error"].lower() or "URL" in out["p"]["error"]


def test_runner_stream_start_pushes_frames_and_stop_confirms(monkeypatch):
    """The full handshake with a fake streamer (no ffmpeg needed): start primes a
    frame and confirms with ok+url; stream_tick feeds frames while due; stop
    confirms on arm.stream.stopped."""
    from ssr_robotics import env_runner as ER

    pushed = []

    class _FakeStreamer:
        def __init__(self, url, w, h, fps=4.0):
            self.url, self.fps, self._active = url, fps, False
        def start(self):
            self._active = True
            return self
        @property
        def active(self):
            return self._active
        def due(self):
            return self._active
        def push(self, data):
            pushed.append(data)
        def stop(self):
            self._active = False

    import ssr_robotics.streamer as S
    monkeypatch.setattr(S, "RtspStreamer", _FakeStreamer)

    bus = MessageBus(name="t", source="t")
    runner = ER.EnvRunner(bus, _StubEnv(), stream_url="rtsp://media:8554/arm",
                          stream_fps=5.0).start()
    started, stopped = threading.Event(), threading.Event()
    out = {}
    bus.subscribe(P.TOPIC_STREAM_STARTED, lambda ev: (out.update(s=ev.payload), started.set()))
    bus.subscribe(P.TOPIC_STREAM_STOPPED, lambda ev: stopped.set())

    bus.publish(P.TOPIC_STREAM_START, {"seq_id": "g2"})
    assert runner.pump(timeout=2.0)
    assert started.wait(2.0)
    assert out["s"]["ok"] is True and out["s"]["url"] == "rtsp://media:8554/arm"
    assert len(pushed) == 1  # the priming frame

    runner.stream_tick()
    assert len(pushed) == 2

    bus.publish(P.TOPIC_STREAM_STOP, {"seq_id": "g2"})
    assert runner.pump(timeout=2.0)
    assert stopped.wait(2.0)
    runner.stream_tick()          # stopped -> no more frames
    assert len(pushed) == 2
    runner.stop()
