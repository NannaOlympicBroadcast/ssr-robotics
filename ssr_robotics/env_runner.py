"""Bridge an environment to the SSR event bus.

:class:`EnvRunner` is bus-agnostic: it accepts any object exposing
``subscribe(pattern, callback)`` and ``publish(topic, payload)`` — both the
in-process :class:`ssr.bus.core.MessageBus` and the remote
:class:`ssr.bus.client.BusClient` satisfy this. It:

* answers ``arm.capabilities.request`` with the env's advertised capabilities
  (so the agent discovers the supported action types / skills at runtime);
* serves ``arm.action.execute`` by invoking the requested skill on the env, then
  publishes a completion event (``arm.grasp.completed`` for grasping skills,
  else ``arm.action.completed``) carrying the result + proprioception + camera
  frame (never object positions — the agent perceives those from the frame);
* answers ``arm.camera.request`` with a fresh camera frame (+ proprioception).

The env must implement: ``reset()``, ``execute(req) -> dict``,
``metrics() -> dict``, ``capabilities() -> dict``, ``frame() -> bytes`` and the
``CAM_W`` / ``CAM_H`` attributes (see :class:`~ssr_robotics.isaac_env.IsaacOpenArmEnv`).

Bus events normally arrive on a background thread (the remote
:class:`ssr.bus.client.BusClient` pumps callbacks from its own asyncio loop
thread). Isaac Sim / Omniverse Kit's simulation context, PhysX and renderer are
**not** thread-safe and may only be driven from the thread that owns
``simulation_app`` — calling ``env.step()``/``env.reset()`` off that thread is a
reliable way to freeze or hang the app. So bus-callback handlers never touch
``self.env`` directly: they enqueue a job and return immediately (keeping the
bus responsive), and the owner of ``simulation_app`` drains the queue by calling
:meth:`pump` in its main loop, running every env call on the correct thread.
"""

from __future__ import annotations

import base64
import queue
import traceback
from typing import Callable

from . import protocol as P

# Skills whose completion is reported on the grasp topic (vs the generic one).
_GRASPING = {"pick"}


class EnvRunner:
    def __init__(self, bus, env, source: str = "openarm-env"):
        self.bus = bus
        self.env = env
        self.source = source
        self._subs: list = []
        self._jobs: "queue.Queue[Callable[[], None]]" = queue.Queue()

    def start(self) -> "EnvRunner":
        self._subs.append(self.bus.subscribe(P.TOPIC_CAPS_REQUEST, self._on_caps))
        self._subs.append(self.bus.subscribe(P.TOPIC_ACTION_EXECUTE, self._on_execute))
        self._subs.append(self.bus.subscribe(P.TOPIC_RESET, self._on_reset))
        self._subs.append(self.bus.subscribe(P.TOPIC_CAMERA_REQUEST, self._on_camera_request))
        # Advertise capabilities once on startup too. start() is called from the
        # main/sim thread before the event loop begins, so this direct call is safe.
        self._publish_caps()
        return self

    def stop(self) -> None:
        for sid in self._subs:
            try:
                self.bus.unsubscribe(sid)
            except Exception:
                pass
        self._subs.clear()

    def pump(self, timeout: float = 0.0) -> bool:
        """Run at most one queued env job. Call this from the thread that owns
        ``simulation_app``, in its main loop. Returns ``True`` if a job ran."""
        try:
            job = self._jobs.get(timeout=timeout)
        except queue.Empty:
            return False
        try:
            job()
        except Exception:
            self._log_exc("queued env job")
        return True

    # ----------------------------------------------------------- publishing
    def _publish(self, topic: str, payload: dict) -> None:
        self.bus.publish(topic, payload)

    @staticmethod
    def _log_exc(label: str) -> None:
        # Some exceptions (asyncio/bare TimeoutError, etc.) stringify to '' —
        # print the type + traceback too, or failures are silently uninformative.
        print(f"[env_runner] {label} failed:\n{traceback.format_exc()}")

    def _frame_fields(self) -> dict:
        # The TiledCamera's render buffer can be momentarily empty right after a
        # reset (before the next step ticks the render pipeline) — don't let a
        # camera hiccup take down the whole state/completion report with it.
        try:
            return {
                "frame_b64": base64.b64encode(self.env.frame()).decode("ascii"),
                "camera": {"width": self.env.CAM_W, "height": self.env.CAM_H},
            }
        except Exception:
            self._log_exc("frame capture")
            return {"camera": {"width": self.env.CAM_W, "height": self.env.CAM_H}}

    def _publish_caps(self) -> None:
        try:
            caps = self.env.capabilities()
            self._publish(P.TOPIC_CAPS, caps)
        except Exception:
            self._log_exc("capabilities publish")

    # ------------------------------------------------------------- handlers
    # These run on a bus-callback thread, never the sim thread: they only ever
    # enqueue a job (cheap, non-blocking) for pump() to run on the right thread.
    def _on_caps(self, ev) -> None:
        self._jobs.put(self._publish_caps)

    def _on_reset(self, ev) -> None:
        # A reset means "stop and start over". Without this, the reset job would sit
        # behind whatever is already queued/running — e.g. a long raw replay holding
        # the sim thread — so the sim looks frozen until that finishes. Interrupt any
        # in-flight multi-step skill and drop the queued backlog so reset runs next.
        interrupt = getattr(self.env, "interrupt", None)
        if callable(interrupt):
            interrupt()
        dropped = self._drain_jobs()
        print(f"[env_runner] received reset (dropped {dropped} queued job(s))")

        def _job() -> None:
            print("[env_runner] reset: running env.reset()")
            try:
                self.env.reset()
                snap = self.env.metrics()
                self._publish(P.TOPIC_CAMERA, {**snap, **self._frame_fields()})
                print("[env_runner] reset: done")
            except Exception:
                self._log_exc("reset")

        self._jobs.put(_job)

    def _drain_jobs(self) -> int:
        """Discard every queued (not-yet-running) job; return how many were dropped."""
        dropped = 0
        while True:
            try:
                self._jobs.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        return dropped

    def _on_camera_request(self, ev) -> None:
        # Fetch a fresh camera frame (+ the arm's own proprioception). This is the
        # only "look" path; it never includes object positions — the agent perceives
        # objects from the frame itself.
        def _job() -> None:
            try:
                snap = self.env.metrics()
                self._publish(P.TOPIC_CAMERA, {**snap, **self._frame_fields()})
            except Exception:
                self._log_exc("camera request")

        self._jobs.put(_job)

    def _on_execute(self, ev) -> None:
        req = P.ArmActionRequest.from_payload(ev.payload)
        # Log every received command on the bridge console. `actions=N` is the
        # single most useful field for debugging "the arm won't move": if a `raw`
        # request arrives with actions=0, the waypoints never left the brain (the
        # brain build is stale / not forwarding them); if N>0 but the arm still
        # doesn't move, the problem is downstream in env.execute.
        print(f"[env_runner] received execute: command={req.command!r} "
              f"seq={req.seq_id} actions={len(req.actions)} args={req.args}")

        def _job() -> None:
            try:
                result = self.env.execute(req)
                frame = self._frame_fields()
            except Exception:
                self._log_exc("execute")
                return
            print(f"[env_runner] execute done: command={req.command!r} "
                  f"seq={req.seq_id} ok={result.get('ok')} error={result.get('error')!r}")
            payload = {"seq_id": req.seq_id, "episode": req.episode,
                       "status": "settled", **result, **frame}
            topic = (P.TOPIC_GRASP_COMPLETED if req.command in _GRASPING
                     else P.TOPIC_ACTION_COMPLETED)
            self._publish(topic, payload)

        self._jobs.put(_job)


def connect_remote(url: str, source: str = "openarm-env", api_key: str | None = None):
    """Return a connected :class:`ssr.bus.client.BusClient`."""
    from ssr.bus import BusClient

    return BusClient(url, source=source, api_key=api_key).connect()
