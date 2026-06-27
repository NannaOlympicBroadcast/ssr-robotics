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

```bash
# 1) brain machine — bus server (the SSR process auto-starts one, or run):
ssr bus serve --host 0.0.0.0 --port 8765

# 2) GPU machine — the bridge (launch via Isaac's python; cameras auto-enabled):
./isaaclab.sh -p scripts/ssr_bridge/run_openarm_bridge.py \
   --bus ws://<brain-host>:8765 --task Isaac-Manip-OpenArm-v0 --headless

# 3) brain machine — drive a natural-language instruction:
ssr arm do "把苹果放到橘子上" --bus-url ws://<brain-host>:8765
```

## Tests (no GPU)

```bash
PYTHONPATH=/path/to/ssr-agent python -m pytest tests/ -q
```

Verifies the shared protocol and that `EnvRunner` advertises capabilities and
serves a skill over the real in-process bus using a stub env. The real
`IsaacOpenArmEnv` is exercised on the robot machine via `ssr-arm-bridge`.
