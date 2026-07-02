# ssr-robotics — Isaac Lab OpenArm ⇄ SSR Agent bus bridge (env side)

The **environment side** of the SSR ⇄ OpenArm integration. It runs next to NVIDIA
Isaac Lab, advertises the arm's capabilities, executes the skills invoked over the
event bus, reports results (objects + camera frame), and pushes the camera as a
live **RTSP stream** when the cerebellum takes over a grasp.

The brain side (NL planning, the suspend/handler/wake loop, the agent tools) and
the **cerebellum** (小脑 — the VLX-Flow realtime grasp loop) live in
**`ssr-agent`** (`ssr/robotics/`, hosted by the `openarm` plugin). This package is
intentionally thin:

```
ssr_robotics/
  protocol.py     # re-exports ssr.robotics.protocol (shared topics + request types)
  isaac_env.py    # IsaacOpenArmEnv: wraps Isaac-Manip-OpenArm-v0, skills via IK
  streamer.py     # RtspStreamer: raw camera frames -> ffmpeg -> RTSP push
  env_runner.py   # EnvRunner: capabilities + skills + camera stream over a bus
  run_bridge.py   # `ssr-arm-bridge` CLI (launches Isaac Sim, connects, serves)
```

## How grasping works (大脑 / 小脑)

The bridge no longer implements a `pick` skill. Grasping is a **closed realtime
loop** owned by the cerebellum in `ssr-agent`:

1. The big brain (SSR agent) crops the target out of the camera frame and
   publishes `arm.grasp.request`.
2. The cerebellum publishes `arm.stream.start`; this bridge starts pushing the
   overhead camera to the configured RTSP address (`--stream-url` /
   `SSR_ARM_STREAM_URL`, needs **ffmpeg** on PATH) and confirms with
   `arm.stream.started`.
3. The cerebellum feeds the stream (via the configured pull URL) + the target
   crop to the Om-Agent **VLX-Flow** model, which emits realtime corrections; each
   arrives here as a `servo` skill (`{dx, dy, dz, grip}` — clamped TCP delta +
   gripper) on `arm.action.execute` and is answered with `arm.action.completed`.
4. On grasp success/failure the cerebellum stops the stream (`arm.stream.stop`)
   and notifies the brain with `arm.grasp.result`.

Run an RTSP server both this machine and the VLX platform can reach (e.g.
[mediamtx](https://github.com/bluenviron/mediamtx)); push and pull can then be
the same URL.

## Grounded action types

Capabilities are **introspected from the live env** (`env.action_manager`), so
they always match the running config. The OpenArm manip env uses end-effector
pose control (`DifferentialInverseKinematicsAction`, body `openarm_hand`) + a
binary gripper. `IsaacOpenArmEnv` implements the object-agnostic skills
(`servo`, `place_at`, `move_above`, `raw`) on top of those primitives, reading
object candidates from the overhead camera (never ground-truth scene state) and
commanding TCP poses via IK.

## Requirements (robot / GPU machine)

* an NVIDIA GPU with **Isaac Sim + Isaac Lab** (`isaaclab`, `torch`, `gymnasium`)
* **`ssr-agent`** installed (provides `ssr.bus` + `ssr.robotics.protocol`)
* the **`openarm`** package from `openarm_isaac_lab` (registers `Isaac-Manip-OpenArm-v0`)
* **`ffmpeg`** on PATH + a reachable RTSP server (for the grasp camera stream)
* `pip install -e .` here (plus `.[isaac]` for `pillow`/`gymnasium`)

`isaaclab` / `torch` / `gymnasium` / `PIL` are imported **lazily**, so this package
imports fine (for protocol/unit tests) without a GPU.

## Run

With a **pip-installed** Isaac Lab (Windows / Linux) there is no `isaaclab.sh` —
the bridge launches Isaac Sim itself, so run it with plain python via the
`ssr-arm-bridge` entry point (equivalently `python -m ssr_robotics.run_bridge`).

```bash
# 1) bus server (any machine; the SSR process also auto-starts an embedded one):
ssr bus serve --host 127.0.0.1 --port 8765

# 2) an RTSP server for the grasp camera stream (e.g. mediamtx on :8554)

# 3) the bridge — pip Isaac Lab: plain python entry point
ssr-arm-bridge --bus ws://127.0.0.1:8765 --task Isaac-Manip-OpenArm-v0 \
    --stream-url rtsp://127.0.0.1:8554/openarm
#    source (git) Isaac Lab instead: ./isaaclab.sh -p .../run_openarm_bridge.py ...

# 4) drive a natural-language instruction on the brain machine
#    (needs GEMINI_API_KEY + ~/.ssr/vlx.json for the cerebellum):
ssr ask --keep-alive 120 "把苹果放到橘子上"
```

Start order: **bus server → RTSP server → bridge → the SSR agent**.

## Tuning grasp reliability

The grasp loop's knobs (all overridable per-instance via `IsaacOpenArmEnv`
constructor args / `SSR_ARM_*` env vars, so no code edit is needed):

* **Servo step size / speed.** `SSR_ARM_SERVO_MAX` (default `0.05` m) clamps each
  VLX correction; `SSR_ARM_SERVO_STEPS` (default `12`) is how many sim steps each
  correction is held. Smaller/fewer = snappier but jerkier.
* **Stream latency.** `SSR_ARM_STREAM_FPS` (default `4`) — the VLX analysis can
  only be as fresh as the stream; the cerebellum's `interval_seconds` /
  `frame_preempt` (in `~/.ssr/vlx.json`) control the analysis cadence.
* **Grip success threshold.** `SSR_ARM_GRIP_EPS` (default `0.005`): summed
  finger-joint width above which a closed gripper is judged to be holding
  something (proprioceptive — no ground-truth object state).
* **Overhead camera too high / wrong angle.** Back-projecting 2-D pixels from a
  high, oblique view carries a large perspective error in the advertised object
  positions. Lower the camera (smaller Z) and zoom it onto the working area; if
  you add a closer camera prim, point the bridge at it with
  `SSR_ARM_CAMERA=<prim_name>` (default `tiled_camera`). Depth-based
  back-projection is preferred over the table-plane fallback.
* **Obstacles with no collision shape.** The arm will drive a straight line through
  the table/stand. Advertise them to the planner with
  `SSR_ARM_OBSTACLES='[{"name":"stand","aabb":[xmin,ymin,zmin,xmax,ymax,zmax]}]'`
  (robot root frame); they surface in `arm.capabilities → obstacles` and the brain
  is prompted to route around them.
* **Waypoints never reached exactly.** Differential-IK never lands precisely on a
  target, so a strict check stalls or false-fails. `SSR_ARM_POS_TOL` (default
  `0.01` m) is the "arrived" radius: once the end-effector TCP is within it the
  move stops early instead of burning steps or over-driving. Widen it if moves are
  judged failed; tighten it for finer placement.

## Tests (no GPU)

```bash
PYTHONPATH=/path/to/ssr-agent python -m pytest tests/ -q
```

Verifies the shared protocol, that `EnvRunner` advertises capabilities, serves a
skill and the camera-stream handshake over the real in-process bus using a stub
env, and the streamer's ffmpeg plumbing. The real `IsaacOpenArmEnv` is exercised
on the robot machine via `ssr-arm-bridge`.
