"""RTSP camera push-streaming for the OpenArm bridge.

When the cerebellum takes over a grasp (``arm.stream.start``), the bridge starts
pushing the overhead camera as a live RTSP stream to a configured address so the
VLX-Flow model can pull and analyse it in realtime. :class:`RtspStreamer` feeds
raw RGB frames to an ``ffmpeg`` child process that encodes H.264
(zero-latency tuning) and publishes to the RTSP URL (typically a mediamtx /
rtsp-simple-server instance both the robot and the VLX platform can reach).

Pure subprocess plumbing — no Isaac / torch imports — so it is unit-testable
without a GPU. Frames must be pushed from the sim thread (the camera buffer is
only safe to read there); the push itself only writes to a pipe and is cheap.
"""

from __future__ import annotations

import shutil
import subprocess
import time


class RtspStreamer:
    """Pushes raw RGB frames to an RTSP URL through ffmpeg."""

    def __init__(self, url: str, width: int, height: int, fps: float = 4.0):
        if not url:
            raise ValueError("RtspStreamer needs a non-empty RTSP push URL")
        self.url = url
        self.width = int(width)
        self.height = int(height)
        self.fps = max(0.5, float(fps))
        self._proc: subprocess.Popen | None = None
        self._last_push = 0.0

    # ------------------------------------------------------------- lifecycle
    def command(self) -> list[str]:
        """The ffmpeg command line (split out for unit tests)."""
        return [
            "ffmpeg", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}", "-r", f"{self.fps:g}",
            "-i", "-",
            "-an",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-g", str(max(1, int(self.fps * 2))),
            "-f", "rtsp", "-rtsp_transport", "tcp", self.url,
        ]

    def start(self) -> "RtspStreamer":
        if self._proc is not None and self._proc.poll() is None:
            return self
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH — required to push the camera RTSP "
                "stream for the cerebellum grasp loop")
        self._proc = subprocess.Popen(
            self.command(), stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        self._last_push = 0.0
        return self

    @property
    def active(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # ---------------------------------------------------------------- frames
    def due(self) -> bool:
        """Whether it is time to push the next frame at the configured fps."""
        return self.active and (time.monotonic() - self._last_push) >= 1.0 / self.fps

    def push(self, rgb_bytes: bytes) -> None:
        """Write one raw RGB24 frame (width*height*3 bytes) to the encoder.

        Raises when the encoder died (e.g. the RTSP server is unreachable) so
        the caller can surface the failure instead of streaming into the void.
        """
        proc = self._proc
        if proc is None or proc.poll() is not None:
            err = b""
            if proc is not None and proc.stderr is not None:
                try:
                    err = proc.stderr.read() or b""
                except Exception:
                    pass
            self._proc = None
            raise RuntimeError(
                f"ffmpeg RTSP push to {self.url} exited: "
                f"{err.decode('utf-8', 'replace').strip() or 'unknown error'}")
        expected = self.width * self.height * 3
        if len(rgb_bytes) != expected:
            raise ValueError(
                f"frame has {len(rgb_bytes)} bytes, expected {expected} "
                f"({self.width}x{self.height}x3 rgb24)")
        proc.stdin.write(rgb_bytes)
        proc.stdin.flush()
        self._last_push = time.monotonic()
