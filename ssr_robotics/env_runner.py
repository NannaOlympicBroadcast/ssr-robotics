"""Bridge an environment to the SSR event bus.

:class:`EnvRunner` is bus-agnostic: it accepts any object exposing
``subscribe(pattern, callback)`` and ``publish(topic, payload)`` — both the
in-process :class:`ssr.bus.core.MessageBus` and the remote
:class:`ssr.bus.client.BusClient` satisfy this. It:

* answers ``arm.capabilities.request`` with the env's advertised capabilities
  (so the agent discovers the supported action types / skills at runtime);
* serves ``arm.action.execute`` by invoking the requested skill on the env in a
  worker thread (so a long motion never blocks the bus), then publishes a
  completion event (``arm.grasp.completed`` for grasping skills, else
  ``arm.action.completed``) carrying the result + scene snapshot + camera frame.

The env must implement: ``reset()``, ``execute(req) -> dict``,
``metrics() -> dict``, ``capabilities() -> dict``, ``frame() -> bytes`` and the
``CAM_W`` / ``CAM_H`` attributes (see :class:`~ssr_robotics.isaac_env.IsaacOpenArmEnv`).
"""

from __future__ import annotations

import base64
import threading

from . import protocol as P

# Skills whose completion is reported on the grasp topic (vs the generic one).
_GRASPING = {"pick"}


class EnvRunner:
    def __init__(self, bus, env, source: str = "openarm-env"):
        self.bus = bus
        self.env = env
        self.source = source
        self._subs: list = []
        self._lock = threading.Lock()  # serialize env stepping (single sim)

    def start(self) -> "EnvRunner":
        self._subs.append(self.bus.subscribe(P.TOPIC_CAPS_REQUEST, self._on_caps))
        self._subs.append(self.bus.subscribe(P.TOPIC_ACTION_EXECUTE, self._on_execute))
        self._subs.append(self.bus.subscribe(P.TOPIC_RESET, self._on_reset))
        self._subs.append(self.bus.subscribe(P.TOPIC_STATE_REQUEST, self._on_state_request))
        # Advertise capabilities once on startup too.
        self._publish_caps()
        return self

    def stop(self) -> None:
        for sid in self._subs:
            try:
                self.bus.unsubscribe(sid)
            except Exception:
                pass
        self._subs.clear()

    # ----------------------------------------------------------- publishing
    def _publish(self, topic: str, payload: dict) -> None:
        self.bus.publish(topic, payload)

    def _frame_fields(self) -> dict:
        # The TiledCamera's render buffer can be momentarily empty right after a
        # reset (before the next step ticks the render pipeline) — don't let a
        # camera hiccup take down the whole state/completion report with it.
        try:
            return {
                "frame_b64": base64.b64encode(self.env.frame()).decode("ascii"),
                "camera": {"width": self.env.CAM_W, "height": self.env.CAM_H},
            }
        except Exception as e:
            print(f"[env_runner] frame capture failed: {e}")
            return {"camera": {"width": self.env.CAM_W, "height": self.env.CAM_H}}

    def _publish_caps(self) -> None:
        try:
            with self._lock:
                caps = self.env.capabilities()
            self._publish(P.TOPIC_CAPS, caps)
        except Exception as e:
            print(f"[env_runner] capabilities publish failed: {e}")

    # ------------------------------------------------------------- handlers
    def _on_caps(self, ev) -> None:
        self._publish_caps()

    def _on_reset(self, ev) -> None:
        try:
            with self._lock:
                self.env.reset()
                snap = self.env.metrics()
            self._publish(P.TOPIC_STATE, {**snap, **self._frame_fields()})
        except Exception as e:
            print(f"[env_runner] reset failed: {e}")

    def _on_state_request(self, ev) -> None:
        try:
            with self._lock:
                snap = self.env.metrics()
            self._publish(P.TOPIC_STATE, {**snap, **self._frame_fields()})
        except Exception as e:
            print(f"[env_runner] state request failed: {e}")

    def _on_execute(self, ev) -> None:
        req = P.ArmActionRequest.from_payload(ev.payload)

        def _run() -> None:
            with self._lock:
                result = self.env.execute(req)
                frame = self._frame_fields()
            payload = {"seq_id": req.seq_id, "episode": req.episode,
                       "status": "settled", **result, **frame}
            topic = (P.TOPIC_GRASP_COMPLETED if req.command in _GRASPING
                     else P.TOPIC_ACTION_COMPLETED)
            self._publish(topic, payload)

        threading.Thread(target=_run, daemon=True).start()


def connect_remote(url: str, source: str = "openarm-env", api_key: str | None = None):
    """Return a connected :class:`ssr.bus.client.BusClient`."""
    from ssr.bus import BusClient

    return BusClient(url, source=source, api_key=api_key).connect()
