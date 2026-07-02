"""``ssr-arm-bridge`` — run the env-side bridge next to Isaac Sim.

Launches the Isaac Sim app, creates the camera-enabled OpenArm manip env, connects
to the SSR bus server and serves ``arm.action.execute`` requests (replying with
``arm.action.completed``: result + metrics + TiledCamera frame) as well as the
cerebellum's ``arm.stream.start``/``stop`` (pushing the camera as a live RTSP
stream to ``--stream-url`` for the VLX-Flow grasp loop — requires ``ffmpeg`` on
PATH and an RTSP server, e.g. mediamtx, that the VLX platform can also reach).

Deployment (on the robot / GPU machine, with ``ssr-agent`` + ``openarm`` installed)::

    # 1. On the brain machine, start a bus server (any ``ssr`` process also starts
    #    an embedded one automatically; this is the explicit standalone form):
    ssr bus serve --host 0.0.0.0 --port 8765

    # 2. On the GPU machine, start the bridge (connects to that bus, serves the
    #    arm.* topics; add --headless to run Isaac without a window). Point
    #    --stream-url at the RTSP server the VLX platform pulls from:
    ssr-arm-bridge --bus ws://<brain-host>:8765 --task Isaac-Manip-OpenArm-v0 \
        --stream-url rtsp://<media-host>:8554/openarm

    # 3. On the brain machine, drive it in natural language. The `arm_*` tools and
    #    the cerebellum (VLX-Flow grasp loop; configure ~/.ssr/vlx.json) come from
    #    the bundled ``openarm`` plugin. The arm runs an async suspend ->
    #    completion -> wake loop, so the brain process must stay alive for it: use
    #    the interactive TUI, or `ssr ask --keep-alive` for a headless one-shot (a
    #    bare `ssr ask` exits before the loop can react).
    ssr                                            # TUI, then type the instruction
    ssr ask --keep-alive 120 "把苹果放到橘子上"     # headless, stays alive 120s idle

This script must launch the Isaac app *before* importing isaaclab env modules, so
all Isaac imports happen inside :func:`main` after ``AppLauncher``.
"""

from __future__ import annotations

import argparse
import faulthandler
import signal


def main(argv: list[str] | None = None) -> int:
    # Ctrl-Break dumps every thread's Python stack to stderr without killing the
    # process — the only way to see where this is stuck when it's hung in our own
    # code (Kit/physx is a native process py-spy can't introspect on Windows).
    # faulthandler.register() itself is Unix-only, so wire SIGBREAK by hand.
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, lambda signum, frame: faulthandler.dump_traceback())

    parser = argparse.ArgumentParser(prog="ssr-arm-bridge", description=__doc__)
    parser.add_argument("--bus", required=True, help="ws:// URL of the SSR bus server")
    parser.add_argument("--api-key", default=None, help="bus API key, if required")
    parser.add_argument("--task", default="Isaac-Manip-OpenArm-v0", help="Isaac Lab gym id")
    parser.add_argument("--stream-url", default=None,
                        help="RTSP push URL for the camera stream the cerebellum's "
                             "VLX-Flow loop pulls (default: SSR_ARM_STREAM_URL)")
    parser.add_argument("--stream-fps", type=float, default=None,
                        help="camera stream frame rate (default: SSR_ARM_STREAM_FPS or 4)")
    # Let Isaac's AppLauncher add its own CLI args (e.g. --headless).
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    except Exception:
        pass
    args = parser.parse_args(argv)
    # The manip env's TiledCamera requires the cameras pipeline.
    args.enable_cameras = True

    # 1. Launch the Isaac Sim app FIRST.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # 2. Now it is safe to import the env adapter (pulls in isaaclab.envs).
    from .env_runner import EnvRunner, connect_remote
    from .isaac_env import IsaacOpenArmEnv

    env = IsaacOpenArmEnv(task=args.task)
    client = connect_remote(args.bus, source="openarm-env", api_key=args.api_key)
    runner = EnvRunner(client, env, stream_url=args.stream_url,
                       stream_fps=args.stream_fps).start()
    print(f"[ssr-arm-bridge] connected to {args.bus}; task={args.task}; "
          f"stream-url={runner.stream_url or '(unset)'}. "
          "Serving arm.action.execute … (Ctrl-C to stop)")

    try:
        # Isaac Sim's sim/render context is not thread-safe and must only be
        # driven from this thread. Bus callbacks run on a different thread (the
        # remote BusClient's own asyncio loop) and only enqueue work; draining it
        # here, on simulation_app's own thread, is what actually steps the env.
        # When idle, poll simulation_app.update() so the app stays responsive
        # (otherwise Kit's UI/render loop never gets pumped and looks frozen).
        # stream_tick() also runs here — the camera buffer is only safe to read
        # on this thread — feeding the RTSP push stream at its configured fps.
        while simulation_app.is_running():
            if not runner.pump(timeout=0.1):
                simulation_app.update()
            runner.stream_tick()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        try:
            client.close()
        except Exception:
            pass
        env.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
