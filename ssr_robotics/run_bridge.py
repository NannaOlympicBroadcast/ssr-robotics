"""``ssr-arm-bridge`` — run the env-side bridge next to Isaac Sim.

Launches the Isaac Sim app, creates the camera-enabled OpenArm lift env, connects
to the SSR bus server and serves ``arm.action.execute`` requests, replying with
``arm.grasp.completed`` events (grasp metrics + TiledCamera frame).

Deployment (on the robot / GPU machine, with ``ssr-agent`` + ``openarm`` installed)::

    # On the brain machine, the SSR process auto-starts an embedded bus server,
    # or run a standalone one:
    ssr bus serve --host 0.0.0.0 --port 8765

    # On the GPU machine, start the bridge:
    ssr-arm-bridge --bus ws://<brain-host>:8765 --task Isaac-Manip-OpenArm-v0

    # Then, on the brain machine, drive an instruction:
    ssr arm do "把苹果放到橘子上" --bus-url ws://<brain-host>:8765

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
    runner = EnvRunner(client, env).start()
    print(f"[ssr-arm-bridge] connected to {args.bus}; task={args.task}. "
          "Serving arm.action.execute … (Ctrl-C to stop)")

    try:
        # Isaac Sim's sim/render context is not thread-safe and must only be
        # driven from this thread. Bus callbacks run on a different thread (the
        # remote BusClient's own asyncio loop) and only enqueue work; draining it
        # here, on simulation_app's own thread, is what actually steps the env.
        # When idle, poll simulation_app.update() so the app stays responsive
        # (otherwise Kit's UI/render loop never gets pumped and looks frozen).
        while simulation_app.is_running():
            if not runner.pump(timeout=0.1):
                simulation_app.update()
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
