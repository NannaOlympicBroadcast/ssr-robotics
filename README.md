# ssr-robotics — Isaac Lab OpenArm ⇄ SSR Agent bus bridge (env side)

The **environment side** of the SSR ⇄ OpenArm integration. It runs next to NVIDIA
Isaac Lab, advertises the arm's capabilities, executes the skills the SSR agent
invokes over the event bus, and reports results (objects + camera frame) so the
agent's bus-handler turns can verify and re-plan.

The brain side (NL planning, the suspend/handler/wake loop, the agent tools) lives
in **`ssr-agent`** (`ssr/robotics/`). This package is intentionally thin:

```
ssr_robotics/
  protocol.py     # re-exports ssr.robotics.protocol (shared topics + request type)
  isaac_env.py    # IsaacOpenArmEnv: wraps Isaac-Manip-OpenArm-v0, skills via IK
  env_runner.py   # EnvRunner: advertises capabilities + serves skills over a bus
  run_bridge.py   # `ssr-arm-bridge` CLI (launches Isaac Sim, connects, serves)
```

## Grounded action types

Capabilities are **introspected from the live env** (`env.action_manager`), so
they always match the running config. The OpenArm manip env uses end-effector
pose control (`DifferentialInverseKinematicsAction`, body `openarm_hand`) + a
binary gripper. `IsaacOpenArmEnv` implements skills (`pick`, `place_on`,
`place_at`, `move_above`, `raw`) on top of those primitives by reading object
poses from the scene and commanding TCP poses via IK.

## Requirements (robot / GPU machine)

* an NVIDIA GPU with **Isaac Sim + Isaac Lab** (`isaaclab`, `torch`, `gymnasium`)
* **`ssr-agent`** installed (provides `ssr.bus` + `ssr.robotics.protocol`)
* the **`openarm`** package from `openarm_isaac_lab` (registers `Isaac-Manip-OpenArm-v0`)
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

# 2) the bridge — pip Isaac Lab: plain python entry point
ssr-arm-bridge --bus ws://127.0.0.1:8765 --task Isaac-Manip-OpenArm-v0
#    source (git) Isaac Lab instead: ./isaaclab.sh -p .../run_openarm_bridge.py ...

# 3) drive a natural-language instruction (needs GEMINI_API_KEY):
ssr arm do "把苹果放到橘子上" --bus-url ws://127.0.0.1:8765
```

Start order: **bus server → bridge → `ssr arm do`**.

## Tuning grasp reliability

Grasping fails most often for four reasons; each has a knob here (all are
`IsaacOpenArmEnv` constructor args, overridable per-instance via `SSR_ARM_*` env
vars, so no code edit is needed):

* **Overhead camera too high / wrong angle.** Back-projecting 2-D pixels from a
  high, oblique view carries a large perspective error, so the arm grasps off to
  the side. The bridge now **repositions the main camera itself** at startup and
  after every reset: a front-side view pulled out beyond the table's front-left
  corner and kept low (default eye `2.0, 1.3, 0.6` looking at `0.32, 0.0, 0.12`,
  relative to the robot's env origin) so it looks *across* the tabletop at a
  shallow, grazing angle (~13° above the surface) and frames the arm and the whole
  table. Move the eye further out / lower it to flatten the angle more, raise it to
  steepen. Tune with `SSR_ARM_CAM_POS` / `SSR_ARM_CAM_TARGET` (`"x,y,z"`), or set
  `SSR_ARM_CAM_POS=keep` to leave the scene's own placement.
  A different camera prim can be selected with `SSR_ARM_CAMERA=<prim_name>`
  (default `tiled_camera`). Depth-based back-projection is already preferred over
  the table-plane fallback.
* **Debug markers polluting the camera.** Isaac Lab's debug visualizations (the
  RGB axis arrows of the IK target / goal frames) render **into the camera
  image**, wrecking both the vision pipeline and the agent looking at the frame.
  The bridge disables every `debug_vis` flag it finds on the env cfg and switches
  the live managers' visualizers off after creation. Set `SSR_ARM_DEBUG_VIS=1` to
  keep the markers for human debugging.
* **Single long-range estimate.** Set `SSR_ARM_WRIST_CAMERA=<prim_name>` to enable
  an **eye-in-hand** correction: `pick` moves above the target from the overhead
  estimate, then re-centres laterally on the object using the wrist camera before
  descending. Off by default (degrades gracefully to overhead-only).
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

## Camera recording (review & reflect)

Any advertised camera (main or wrist) can be recorded over the bus:
`arm.record.start` `{camera, every?, max_frames?}` begins buffering every Nth sim
step's frame (defaults: every 5th step, 500-frame cap — overruns are counted and
reported, never silent); `arm.record.stop` encodes the buffer (MP4 via
imageio+ffmpeg when installed, else an animated GIF via Pillow) and publishes it
on `arm.record.saved` (`video_b64`, `format`, `fps`, `frames`, `dropped`). The
brain-side `arm_record_start` / `arm_record_stop` tools wrap these and save the
video locally so the agent can **review the whole motion and reflect** — far more
signal than a single after-the-fact frame.

## Tests (no GPU)

```bash
PYTHONPATH=/path/to/ssr-agent python -m pytest tests/ -q
```

Verifies the shared protocol and that `EnvRunner` advertises capabilities and
serves a skill over the real in-process bus using a stub env. The real
`IsaacOpenArmEnv` is exercised on the robot machine via `ssr-arm-bridge`.
