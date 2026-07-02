"""Unit tests for the RTSP push streamer (command construction + guards only —
actually pushing needs ffmpeg + an RTSP server, exercised in the E2E run)."""

from __future__ import annotations

import pytest

from ssr_robotics.streamer import RtspStreamer


def test_command_encodes_rawvideo_to_rtsp():
    s = RtspStreamer("rtsp://media:8554/arm", 320, 240, fps=5)
    cmd = s.command()
    assert cmd[0] == "ffmpeg"
    assert "rawvideo" in cmd and "rgb24" in cmd
    assert "320x240" in cmd
    assert "zerolatency" in cmd
    assert cmd[-1] == "rtsp://media:8554/arm"
    assert "-rtsp_transport" in cmd and "tcp" in cmd


def test_requires_url():
    with pytest.raises(ValueError):
        RtspStreamer("", 320, 240)


def test_push_rejects_wrong_frame_size():
    s = RtspStreamer("rtsp://media:8554/arm", 4, 2, fps=2)

    class _P:  # a fake live ffmpeg process
        stdin = None
        stderr = None
        def poll(self):
            return None

    s._proc = _P()
    with pytest.raises(ValueError):
        s.push(b"\x00" * 5)  # expected 4*2*3 = 24 bytes


def test_push_raises_when_encoder_died():
    s = RtspStreamer("rtsp://media:8554/arm", 4, 2, fps=2)
    with pytest.raises(RuntimeError):
        s.push(b"\x00" * 24)  # never started


def test_due_only_when_active():
    s = RtspStreamer("rtsp://media:8554/arm", 4, 2, fps=2)
    assert s.due() is False
