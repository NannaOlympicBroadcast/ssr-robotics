"""Bridge an environment to the SSR event bus.

:class:`EnvRunner` is bus-agnostic: it accepts any object exposing
``subscribe(pattern, callback)`` and ``publish(topic, payload)`` — both the
in-process :class:`ssr.bus.core.MessageBus` and the remote
:class:`ssr.bus.client.BusClient` satisfy this. It:

* answers ``arm.capabilities.request`` with the env's advertised capabilities
  (so the agent discovers the supported action types / skills at runtime);
* serves ``arm.action.execute`` by invoking the requested skill on the env, then
  publishes an ``arm.action.completed`` event carrying the result + scene
  snapshot + camera frame;
* serves ``arm.stream.start`` / ``arm.stream.stop`` from the cerebellum by
  pushing the camera as a live RTSP stream (see :mod:`ssr_robotics.streamer`)
  to the configured push address (``--stream-url`` / ``SSR_ARM_STREAM_URL``),
  confirming with ``arm.stream.started`` / ``arm.stream.stopped``. The owner of
  the sim thread must call :meth:`EnvRunner.stream_tick` in its main loop to
  feed frames at the configured fps.

The env must implement: ``reset()``, ``execute(req) -> dict``,
``metrics() -> dict``, ``capabilities() -> dict``, ``frame() -> bytes``,
``frame_rgb() -> bytes`` and the ``CAM_W`` / ``CAM_H`` attributes (see
:class:`~ssr_robotics.isaac_env.IsaacOpenArmEnv`).

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
import os
import queue
import traceback
from typing import Callable

from . import protocol as P


class EnvRunner:
    def __init__(self, bus, env, source: str = "openarm-env",
                 stream_url: str | None = None, stream_fps: float | None = None):
        self.bus = bus
        self.env = env
        self.source = source
        # Where the camera RTSP stream is pushed when the cerebellum asks for it
        # (arm.stream.start). Constructor arg wins, else SSR_ARM_STREAM_URL.
        self.stream_url = (stream_url if stream_url is not None
                           else os.environ.get("SSR_ARM_STREAM_URL", "")).strip()
        raw_fps = os.environ.get("SSR_ARM_STREAM_FPS", "")
        try:
            env_fps = float(raw_fps) if raw_fps.strip() else 4.0
        except ValueError:
            env_fps = 4.0
        self.stream_fps = env_fps if stream_fps is None else float(stream_fps)
        self._streamer = None
        self._subs: list = []
        self._jobs: "queue.Queue[Callable[[], None]]" = queue.Queue()

    def start(self) -> "EnvRunner":
        self._subs.append(self.bus.subscribe(P.TOPIC_CAPS_REQUEST, self._on_caps))
        self._subs.append(self.bus.subscribe(P.TOPIC_ACTION_EXECUTE, self._on_execute))
        self._subs.append(self.bus.subscribe(P.TOPIC_RESET, self._on_reset))
        self._subs.append(self.bus.subscribe(P.TOPIC_STATE_REQUEST, self._on_state_request))
        self._subs.append(self.bus.subscribe(P.TOPIC_STREAM_START, self._on_stream_start))
        self._subs.append(self.bus.subscribe(P.TOPIC_STREAM_STOP, self._on_stream_stop))
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
        self._stop_streamer()

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
                self._publish(P.TOPIC_STATE, {**snap, **self._frame_fields()})
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

    def _on_state_request(self, ev) -> None:
        def _job() -> None:
            try:
                snap = self.env.metrics()
                self._publish(P.TOPIC_STATE, {**snap, **self._frame_fields()})
            except Exception:
                self._log_exc("state request")

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
            self._publish(P.TOPIC_ACTION_COMPLETED, payload)

        self._jobs.put(_job)

    # ------------------------------------------------------- camera streaming
    # The cerebellum's grasp loop needs the live camera as an RTSP stream the
    # VLX platform can pull. Stream start/stop are bus-triggered; the actual
    # frame capture happens in stream_tick() on the sim thread.
    def _on_stream_start(self, ev) -> None:
        seq_id = (ev.payload or {}).get("seq_id", "")

        def _job() -> None:
            if not self.stream_url:
                self._publish(P.TOPIC_STREAM_STARTED, {
                    "seq_id": seq_id, "ok": False,
                    "error": "no RTSP push URL configured on the bridge "
                             "(--stream-url / SSR_ARM_STREAM_URL)"})
                return
            try:
                if self._streamer is None or not self._streamer.active:
                    from .streamer import RtspStreamer

                    self._streamer = RtspStreamer(
                        self.stream_url, self.env.CAM_W, self.env.CAM_H,
                        fps=self.stream_fps).start()
                # Prime the encoder with the current frame right away so the
                # stream is immediately pullable.
                self._streamer.push(self.env.frame_rgb())
                print(f"[env_runner] camera stream -> {self.stream_url} "
                      f"({self.env.CAM_W}x{self.env.CAM_H}@{self.stream_fps:g}fps)")
                self._publish(P.TOPIC_STREAM_STARTED, {
                    "seq_id": seq_id, "ok": True, "url": self.stream_url,
                    "fps": self.stream_fps})
            except Exception as e:
                self._log_exc("stream start")
                self._stop_streamer()
                self._publish(P.TOPIC_STREAM_STARTED, {
                    "seq_id": seq_id, "ok": False, "error": str(e)})

        self._jobs.put(_job)

    def _on_stream_stop(self, ev) -> None:
        seq_id = (ev.payload or {}).get("seq_id", "")

        def _job() -> None:
            self._stop_streamer()
            print("[env_runner] camera stream stopped")
            self._publish(P.TOPIC_STREAM_STOPPED, {"seq_id": seq_id, "ok": True})

        self._jobs.put(_job)

    def _stop_streamer(self) -> None:
        streamer, self._streamer = self._streamer, None
        if streamer is not None:
            try:
                streamer.stop()
            except Exception:
                pass

    def stream_tick(self) -> None:
        """Push the next camera frame if streaming is active and due.

        MUST be called from the thread that owns ``simulation_app`` (the same
        one that calls :meth:`pump`) — the camera buffer is only safe to read
        there. Cheap when idle: a monotonic-clock check and nothing else."""
        streamer = self._streamer
        if streamer is None or not streamer.due():
            return
        try:
            streamer.push(self.env.frame_rgb())
        except Exception:
            self._log_exc("stream frame push")
            self._stop_streamer()


def connect_remote(url: str, source: str = "openarm-env", api_key: str | None = None):
    """Return a connected :class:`ssr.bus.client.BusClient`."""
    from ssr.bus import BusClient

    return BusClient(url, source=source, api_key=api_key).connect()
