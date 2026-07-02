"""Adapter for the real Isaac Lab OpenArm manipulation environment.

Wraps ``Isaac-Manip-OpenArm-v0`` (the IK + camera + apple/orange variant added in
``openarm_isaac_lab``) behind the env interface :class:`EnvRunner` expects:
``reset()``, ``execute(req)``, ``metrics()``, ``frame()``, ``capabilities()`` and
``CAM_W`` / ``CAM_H``.

**Capabilities are introspected from the live env**, not hardcoded: the action
space is read from ``env.action_manager`` (term names, dims, classes). Objects are
**never** read from ``env.scene`` ground truth — they are perceived only through
the camera (see :meth:`IsaacOpenArmEnv._perceive`), so the bridge generalizes to a
real RGBD camera with no privileged scene knowledge. The arm's real action types — grounded
in the OpenArm repo — are end-effector pose (``DifferentialInverseKinematicsAction``
on body ``openarm_hand``) plus a binary gripper (``openarm_finger_joint.*``). On
top of those primitives the bridge implements a few **object-agnostic** skills
(servo / place_at / move_above / raw) the agent may invoke; the agent discovers
them via ``arm_describe`` and plans accordingly. The bridge knows nothing about
*what* it manipulates — every skill is driven by world coordinates the agent
supplies from its own perception of the camera frame, never by a hardcoded object
name or identity — so it generalizes beyond any particular scene.

**Grasping is not a bridge skill any more.** It is closed-loop-driven by the
cerebellum (``ssr.robotics.cerebellum`` + the VLX-Flow streaming vision model):
the bridge only pushes its camera as an RTSP stream (see
:mod:`ssr_robotics.streamer`) and executes the cerebellum's stream of small
``servo`` corrections (TCP deltas + gripper) in realtime.

The Isaac Sim app MUST be launched (``isaaclab.app.AppLauncher``) before
constructing this class — see ``run_bridge.py`` / ``openarm_isaac_lab/scripts``.
"""

from __future__ import annotations

import math
import threading
import traceback

from . import protocol as P
from . import vision as V

# Skills the bridge implements on top of the arm's primitive action types. They are
# deliberately object-agnostic — the bridge knows nothing about *what* it is
# manipulating, so this generalizes beyond any particular scene. Coordinates are in
# the robot ROOT frame; the agent gets candidate object positions from the vision
# snapshot (arm_describe / arm_get_scene "objects", detected by camera) and never
# refers to objects by a hardcoded name/identity.
SKILLS = [
    {"name": "servo", "args": {"dx": "float", "dy": "float", "dz": "float",
                               "grip": "open|close|hold"},
     "desc": "small realtime end-effector correction (root-frame deltas in "
             "metres, clamped) plus an optional gripper actuation — the "
             "primitive the cerebellum's VLX-Flow grasp loop drives"},
    {"name": "place_at", "args": {"x": "float", "y": "float"},
     "desc": "place the currently-held object at root-frame (x, y)"},
    {"name": "move_above", "args": {"x": "float", "y": "float"},
     "desc": "move the end-effector above root-frame (x, y)"},
    {"name": "raw", "args": {"actions": "list[float vectors]"},
     "desc": "replay low-level action vectors matching action_space.dof"},
]

# Defaults for the tunable manipulation parameters below. They are *defaults* only:
# every one is overridable per-instance via the IsaacOpenArmEnv constructor (and the
# constructor reads SSR_ARM_* env vars), so nothing here is baked in — a different
# gripper, object size or surface height needs no code edit.

# Default grasp orientation (quaternion wxyz) in the robot root frame (top-down).
GRASP_QUAT = (0.0, 1.0, 0.0, 0.0)

APPROACH_Z = 0.12   # hover height above a target before descending (m)
# TCP height relative to the target point when releasing (m).
GRASP_Z = 0.0

# Per-command clamp for one cerebellum `servo` correction (m). Keeps a bad model
# output from throwing the arm across the workspace; the realtime loop converges
# by issuing many small corrections instead.
SERVO_MAX = 0.05
# Sim steps to hold each servo target — small on purpose: a servo correction is
# meant to be quick so the next VLX-Flow frame sees its effect promptly.
SERVO_STEPS = 12

# World height (m) of the plane the camera back-projects onto when locating an
# object by vision (used only when no depth is available) — the approximate object
# resting height on the table — and the default target z for a place/move when only
# x, y are given.
TABLE_Z = 0.055

# Chroma threshold for the vision foreground/background split (see vision.py).
CHROMA_THRESH = 45.0

# Summed finger-joint width above which the gripper is judged to be holding
# something after a close (an object wedged the fingers open). Proprioceptive, so
# grasp success needs no ground-truth object state. Tune on the real robot.
GRIP_HOLD_EPS = 0.005

# Waypoint position tolerance (m): once the end-effector TCP is within this of a
# commanded pose, the move is judged "arrived" and stops early instead of burning
# the remaining fixed steps (or over-driving past it). Differential-IK never lands
# *exactly* on a target, so requiring exactness would either waste steps or, with
# too few of them, report a phantom failure on a move that is physically fine. A
# small tolerance is what makes multi-waypoint motion reliable. Tune on the robot.
POS_TOL = 0.01

# Default overhead camera prim name in the scene. Overridable (SSR_ARM_CAMERA /
# constructor) so a re-positioned or lowered/zoomed camera — which sharply reduces
# the perspective error when back-projecting pixels to 3-D — can be selected with
# no code change. This is also the camera the RTSP push stream reads.
CAMERA_NAME = "tiled_camera"


def _env_float(name: str, default: float) -> float:
    """Read a float override from the environment, ignoring blank/garbage values."""
    import os

    raw = os.environ.get(name, "")
    try:
        return float(raw) if raw.strip() else default
    except (TypeError, ValueError):
        return default


def _within_tol(current, target, tol: float) -> bool:
    """Whether 3-D point ``current`` is within ``tol`` metres (Euclidean) of
    ``target``.

    Used to decide a commanded waypoint has actually been reached. ``current`` may
    be ``None`` (the TCP could not be read) → treated as *not* reached, so motion
    safely falls back to running its fixed step budget. A non-positive ``tol``
    disables the early-exit entirely (full fixed steps run)."""
    if current is None or tol is None or tol <= 0:
        return False
    return sum((float(a) - float(b)) ** 2 for a, b in zip(current, target)) <= tol * tol


def _parse_obstacles(raw) -> list:
    """Parse the ``SSR_ARM_OBSTACLES`` JSON into a list of obstacle descriptors.

    Obstacles are *advertised* to the brain (in :meth:`capabilities`) so its planner
    can treat them as collision bodies to route around — the sim deliberately leaves
    some supports without a collision shape, so the arm will happily drive a
    straight line through the table/stand unless the planner is told not to. Expected
    JSON: a list of ``{"name": str, "aabb": [xmin, ymin, zmin, xmax, ymax, zmax]}``
    in the robot ROOT frame. Returns ``[]`` on blank/garbage rather than raising —
    obstacles are optional metadata, never required for the bridge to run."""
    import json

    if not raw or not str(raw).strip():
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict) or "aabb" not in item:
            continue
        try:
            aabb = [float(c) for c in item["aabb"]]
        except (TypeError, ValueError):
            continue
        if len(aabb) != 6:
            continue
        out.append({"name": str(item.get("name") or f"obstacle{len(out)}"), "aabb": aabb})
    return out


class IsaacOpenArmEnv:
    def __init__(self, task: str = "Isaac-Manip-OpenArm-v0", steps_per_move: int = 30,
                 settle_steps: int = 15, steps_per_waypoint: int = 10, num_envs: int = 1,
                 grasp_quat=GRASP_QUAT, approach_z: float | None = None,
                 grasp_z: float | None = None,
                 table_z: float | None = None, chroma_thresh: float | None = None,
                 grip_hold_eps: float | None = None, pos_tol: float | None = None,
                 servo_max: float | None = None, servo_steps: int | None = None,
                 camera: str | None = None,
                 obstacles=None):
        import os

        import gymnasium as gym
        import torch

        import openarm.tasks  # noqa: F401  registers the gym ids
        from isaaclab.utils.math import subtract_frame_transforms
        from isaaclab_tasks.utils import parse_env_cfg

        self.torch = torch
        self._sub = subtract_frame_transforms
        self.task = task
        self.steps_per_move = steps_per_move
        self.settle_steps = settle_steps
        # Sim steps to hold each `raw` waypoint. The action space is differential-IK
        # pose targets, so — exactly like _goto — each target needs several steps for
        # the controller to actually drive the arm there. Stepping once per waypoint
        # moves the arm imperceptibly (it never reaches any pose), so a trajectory
        # looks like no motion at all.
        self.steps_per_waypoint = steps_per_waypoint
        # Tunable manipulation parameters — constructor arg wins, else SSR_ARM_* env
        # var, else the module default. None of these are hardcoded into the logic.
        self.grasp_quat = tuple(grasp_quat)
        self.approach_z = _env_float("SSR_ARM_APPROACH_Z", APPROACH_Z) if approach_z is None else float(approach_z)
        self.grasp_z = _env_float("SSR_ARM_GRASP_Z", GRASP_Z) if grasp_z is None else float(grasp_z)
        self.table_z = _env_float("SSR_ARM_TABLE_Z", TABLE_Z) if table_z is None else float(table_z)
        self.chroma_thresh = _env_float("SSR_ARM_CHROMA", CHROMA_THRESH) if chroma_thresh is None else float(chroma_thresh)
        self.grip_hold_eps = _env_float("SSR_ARM_GRIP_EPS", GRIP_HOLD_EPS) if grip_hold_eps is None else float(grip_hold_eps)
        # Waypoint arrival tolerance — once the TCP is this close to a commanded pose
        # the move stops early (see _goto). Constructor arg wins, else SSR_ARM_POS_TOL.
        self.pos_tol = _env_float("SSR_ARM_POS_TOL", POS_TOL) if pos_tol is None else float(pos_tol)
        # Cerebellum servo tunables: per-command delta clamp + steps per correction.
        self.servo_max = _env_float("SSR_ARM_SERVO_MAX", SERVO_MAX) if servo_max is None else float(servo_max)
        self.servo_steps = (int(_env_float("SSR_ARM_SERVO_STEPS", SERVO_STEPS))
                            if servo_steps is None else int(servo_steps))
        # Which scene camera the overhead perception (and the RTSP stream) reads.
        self.camera_name = camera or os.environ.get("SSR_ARM_CAMERA") or CAMERA_NAME
        # Obstacles advertised to the brain's planner as collision bodies to avoid.
        self.obstacles = (_parse_obstacles(os.environ.get("SSR_ARM_OBSTACLES", ""))
                          if obstacles is None else list(obstacles))
        # The `env_cfg_entry_point` gym.make() is registered with is inert metadata —
        # Isaac Lab requires resolving it into a real cfg instance first. This also
        # forces num_envs=1: the task's registered default is RL-training scale
        # (e.g. 4096), but this bridge drives a single real/simulated arm.
        env_cfg = parse_env_cfg(task, num_envs=num_envs)
        self.env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
        self.device = self.env.unwrapped.device
        cam = self.env.unwrapped.scene[self.camera_name]
        self.CAM_H, self.CAM_W = int(cam.image_shape[0]), int(cam.image_shape[1])
        self._holding: bool = False  # whether a grasp is currently held
        self._gripper_open: bool = True  # the commanded gripper state (servo loop)
        # Perception cache: vision is recomputed lazily and only after the scene has
        # actually moved (a sim step invalidates it), so reporting state/capabilities
        # back-to-back doesn't re-run the camera pipeline several times per request.
        self._perceive_cache: list[dict] | None = None
        # Fixed camera calibration (intrinsics + world pose), captured once on first
        # use — like a real hand-eye calibration — so object localization never
        # queries the simulator for where things are; see _camera_calibration().
        self._cam_calib: dict | None = None
        # End-effector body name (for reading the live TCP position to judge waypoint
        # arrival), resolved lazily from the IK action term and cached. "" once
        # resolution has been attempted and failed, so we don't retry every move.
        self._ee_body: str | None = None
        # Set from another thread (the bus-callback handler on reset) to ask a
        # running multi-step skill to bail out, so a reset doesn't have to wait for
        # a long raw replay to finish before it can run.
        self._interrupt = threading.Event()
        self.reset()

    def _step(self, action):
        """Advance the sim one step and invalidate the perception cache (the scene
        moved, so any cached detection is now stale)."""
        self._perceive_cache = None
        return self.env.step(action)

    def interrupt(self) -> None:
        """Ask an in-flight multi-step skill (e.g. a raw replay) to stop ASAP.

        Checked between waypoints, so it cannot break out of a single hung
        ``env.step``, but it lets a reset preempt a long/runaway replay."""
        self._interrupt.set()

    # ------------------------------------------------------------------ env
    def reset(self, target: str | None = None) -> None:
        self._interrupt.clear()
        self._perceive_cache = None
        self.env.reset()
        self._holding = False
        self._gripper_open = True

    def _scene(self):
        return self.env.unwrapped.scene

    @property
    def _num(self) -> int:
        return self.env.unwrapped.num_envs

    # ------------------------------------------------------------ perception
    def _calib_of(self, cam) -> dict:
        """Intrinsics ``K`` + world pose for a camera object (NumPy)."""
        from isaaclab.utils.math import matrix_from_quat

        return {
            "K": cam.data.intrinsic_matrices[0].detach().cpu().numpy(),
            "pos": cam.data.pos_w[0].detach().cpu().numpy(),
            "rot": matrix_from_quat(cam.data.quat_w_ros[0]).detach().cpu().numpy(),
        }

    def _camera_calibration(self) -> dict:
        """The overhead camera's intrinsics ``K`` + world pose, captured ONCE and
        cached — the analogue of a real camera's one-time (hand-eye) calibration.

        Reading these here, once, is the *only* time the camera's geometry is taken
        from the simulator; every subsequent perception uses just the live image and
        this fixed calibration, so the bridge never asks the simulator *where things
        are*. The overhead camera is static, so a single capture is exact."""
        if self._cam_calib is None:
            self._cam_calib = self._calib_of(self._scene()[self.camera_name])
        return self._cam_calib

    def _detect_from(self, cam, calib) -> list[dict]:
        """Run the vision pipeline on one camera and return root-frame detections.

        Used by the overhead view (:meth:`_perceive`). The ONLY inputs are the live
        image (+ depth) and the supplied calibration; the robot's base pose is used
        purely to express the result in its control frame. NO object/scene
        ground-truth is read. Each blob
        is back-projected with the camera's per-pixel **depth** when available
        (``distance_to_image_plane`` — exact 3-D, robust to differing object height)
        and falls back to the ``table_z`` plane intersection only when depth is not.
        """
        out = cam.data.output
        rgb = out["rgb"][0].detach().cpu().numpy()[..., :3].astype("float32")
        depth = None
        if "distance_to_image_plane" in out:
            depth = out["distance_to_image_plane"][0].detach().cpu().numpy()
            if depth.ndim == 3:
                depth = depth[..., 0]
        blobs = V.connected_blobs(V.foreground_mask(rgb, self.chroma_thresh))
        K, cam_pos, cam_rot = calib["K"], calib["pos"], calib["rot"]
        t = self.torch
        robot = self._scene()["robot"]
        dets = []
        for (u, v, n) in blobs:
            pw, src = None, "table"
            if depth is not None:
                vi, ui = int(round(v)), int(round(u))
                if 0 <= vi < depth.shape[0] and 0 <= ui < depth.shape[1]:
                    pw = V.pixel_to_point_with_depth(u, v, float(depth[vi, ui]),
                                                     K, cam_pos, cam_rot)
                    if pw is not None:
                        src = "depth"
            if pw is None:
                pw = V.pixel_to_table_point(u, v, K, cam_pos, cam_rot, self.table_z)
            if pw is None:
                continue
            pwt = t.tensor([pw], dtype=t.float32, device=self.device)
            pb, _ = self._sub(robot.data.root_pos_w, robot.data.root_quat_w, pwt)
            dets.append({
                "pixel": [round(float(u), 1), round(float(v), 1)], "pixels": int(n),
                "src": src,
                "world": [round(float(c), 4) for c in pw],
                "root": [round(float(pb[0, 0]), 4), round(float(pb[0, 1]), 4),
                         round(float(pb[0, 2]), 4)],
            })
        return dets

    def _perceive(self) -> list[dict]:
        """Detect graspable objects BY VISION from the overhead camera.

        Returns a list (largest blob first) of detections with the centroid
        ``pixel``, blob ``pixels`` count, and the back-projected ``world`` +
        robot-``root`` positions. Cached until the next sim step (:meth:`_step`) so
        repeated state/cap reports don't re-run the pipeline.
        """
        if self._perceive_cache is not None:
            return self._perceive_cache

        dets = self._detect_from(self._scene()[self.camera_name], self._camera_calibration())
        self._perceive_cache = dets
        print(f"[isaac_env] vision: detected {len(dets)} object(s): "
              f"{[(d['pixel'], d['src'], d['root']) for d in dets]}")
        return dets

    def objects_world(self) -> dict:
        """Perceived object positions (robot ROOT frame) BY VISION, keyed generically
        (obj0, obj1, … largest blob first). No ground-truth scene state — this is the
        situational awareness the agent uses to choose target coordinates."""
        return {f"obj{i}": d["root"] for i, d in enumerate(self._perceive())}

    # --------------------------------------------------------- IK stepping
    def _ik_action(self, pos, gripper_open: bool):
        """Build an [px,py,pz,qw,qx,qy,qz, gripper] action for all envs."""
        t = self.torch
        g = P.GRIPPER_OPEN if gripper_open else P.GRIPPER_CLOSE
        vec = [float(pos[0]), float(pos[1]), float(pos[2]), *self.grasp_quat, g]
        return t.tensor([vec] * self._num, dtype=t.float32, device=self.device)

    def _ee_body_name(self) -> str | None:
        """Name of the controlled end-effector body, read once from the IK action
        term's cfg (same source :meth:`capabilities` advertises) and cached."""
        if self._ee_body is not None:
            return self._ee_body or None
        name = ""
        try:
            am = self.env.unwrapped.action_manager
            for term_name in list(am.active_terms):
                term = am.get_term(term_name) if hasattr(am, "get_term") else None
                cfg = getattr(term, "cfg", None)
                body = getattr(cfg, "body_name", None) if cfg is not None else None
                if body:
                    name = body
                    break
        except Exception:
            name = ""
        self._ee_body = name
        return name or None

    def _tcp_pos(self) -> list | None:
        """Live end-effector position in the robot ROOT frame, or ``None`` if it
        can't be read.

        This is proprioception (the arm's own forward kinematics), the same category
        as gripper width — NOT object/scene ground truth — and is used only to tell
        whether a commanded waypoint has been reached (see :meth:`_goto`)."""
        body = self._ee_body_name()
        if not body:
            return None
        try:
            robot = self._scene()["robot"]
            names = list(robot.data.body_names)
            if body not in names:
                return None
            idx = names.index(body)
            pw = robot.data.body_pos_w[:, idx, :]
            pb, _ = self._sub(robot.data.root_pos_w, robot.data.root_quat_w, pw)
            return [float(pb[0, 0]), float(pb[0, 1]), float(pb[0, 2])]
        except Exception:
            return None

    def _goto(self, pos, gripper_open: bool, steps: int | None = None,
              tol: float | None = None) -> bool:
        """Drive the TCP toward ``pos``, stopping early once within ``tol`` of it.

        Differential-IK converges over several steps, so each waypoint is held for
        up to ``steps`` (default :attr:`steps_per_move`) sim steps — but as soon as
        the TCP is within ``tol`` metres (default :attr:`pos_tol`) the move returns,
        rather than wasting the rest of the budget or over-driving past the target.
        Pass ``tol=0`` to disable the early-exit (e.g. the gripper open/close settle,
        which isn't a TCP move). Returns whether the target was reached within
        tolerance (always ``False`` when the early-exit is disabled or the TCP is
        unreadable — callers that don't care just ignore it)."""
        action = self._ik_action(pos, gripper_open)
        tol = self.pos_tol if tol is None else tol
        reached = False
        for _ in range(steps or self.steps_per_move):
            self._step(action)
            if tol and tol > 0 and _within_tol(self._tcp_pos(), pos, tol):
                reached = True
                break
        return reached

    # -------------------------------------------------------------- skills
    def execute(self, req: "P.ArmActionRequest") -> dict:
        self._interrupt.clear()  # fresh command — drop any stale interrupt
        try:
            ok = self._run_skill(req)
            return self._report(req.command, ok=ok)
        except Exception as e:  # report failures rather than crashing the bridge
            return self._report(req.command, ok=False, error=str(e))

    def _run_skill(self, req: "P.ArmActionRequest") -> bool:
        cmd, args = req.command, req.args
        if cmd == "raw":
            t = self.torch
            # Accept the waypoints from either the dedicated `actions` field (used by
            # arm_act) or args["actions"] — which is how the capability descriptor
            # advertises the raw skill (SKILLS: raw args {"actions": ...}), so the
            # agent legitimately puts them there via arm_invoke('raw', {...}). Reading
            # both keeps the bridge working regardless of how the brain forwards them.
            actions = req.actions or (args.get("actions") if isinstance(args, dict) else None) or []
            if not isinstance(actions, list) or not actions:
                # An empty action list used to "succeed" silently (the loop ran zero
                # times), so the arm never moved yet the step reported ok. Make it a
                # loud error instead of silent stillness.
                raise ValueError("raw skill received no action vectors")
            if not req.actions:
                print("[isaac_env] raw: actions read from args['actions'] "
                      "(dedicated 'actions' field was empty)")
            am = self.env.unwrapped.action_manager
            # total_action_dim isn't guaranteed across isaaclab versions (capabilities()
            # reads it defensively too); fall back to the first vector's width rather
            # than risk an AttributeError that would abort the whole replay.
            dof = int(getattr(am, "total_action_dim", 0)) or len(actions[0])
            steps = max(1, int(self.steps_per_waypoint))
            print(f"[isaac_env] raw: replaying {len(actions)} waypoints "
                  f"x {steps} steps (dof={dof})")
            for i, vec in enumerate(actions):
                if self._interrupt.is_set():
                    print(f"[isaac_env] raw: interrupted at waypoint {i}/{len(actions)}")
                    return False
                if dof and len(vec) != dof:
                    raise ValueError(f"raw action[{i}] has {len(vec)} dims, expected {dof}")
                if not all(math.isfinite(v) for v in vec):
                    raise ValueError(f"raw action[{i}] has a non-finite value: {vec}")
                a = t.tensor([vec] * self._num, dtype=t.float32, device=self.device)
                for _ in range(steps):
                    self._step(a)
            print(f"[isaac_env] raw: done ({len(actions)} waypoints)")
            return True
        if cmd == "move_above":
            pos = self._target_xyz(args)
            pos[2] = pos[2] + self.approach_z
            self._goto(pos, gripper_open=self._gripper_open)
            return True
        if cmd == "servo":
            return self._servo(args)
        if cmd == "place_at":
            return self._place(self._target_xyz(args))
        raise ValueError(f"unknown skill '{cmd}'")

    def _target_xyz(self, args: dict):
        """Resolve a coordinate target to a robot-root-frame [x, y, z] list.

        Object-agnostic: the agent supplies the coordinates (from its own
        perception of the camera frame), the bridge just executes them."""
        if "x" in args and "y" in args:
            z = float(args["z"]) if args.get("z") is not None else self.table_z
            return [float(args["x"]), float(args["y"]), z]
        raise ValueError("target needs 'x' and 'y' (and optionally 'z')")

    def _servo(self, args: dict) -> bool:
        """One realtime correction from the cerebellum's VLX-Flow loop.

        Displaces the TCP by a small root-frame delta (each component clamped to
        ``±servo_max``) and/or actuates the gripper (``grip``: open/close/hold).
        Motion runs first with the currently-commanded gripper state; a gripper
        change then settles in place. Grasp success stays proprioceptive: after a
        close, an object wedging the fingers open sets ``holding``."""
        def _delta(key: str) -> float:
            try:
                v = float(args.get(key) or 0.0)
            except (TypeError, ValueError):
                raise ValueError(f"servo: '{key}' must be a number")
            if not math.isfinite(v):
                raise ValueError(f"servo: '{key}' is not finite")
            return max(-self.servo_max, min(self.servo_max, v))

        dx, dy, dz = _delta("dx"), _delta("dy"), _delta("dz")
        grip = str(args.get("grip") or "hold").strip().lower()
        if grip not in ("open", "close", "hold"):
            raise ValueError(f"servo: grip must be open|close|hold, got '{grip}'")
        cur = self._tcp_pos()
        if cur is None:
            raise ValueError("servo: cannot read the end-effector position")
        target = [cur[0] + dx, cur[1] + dy, cur[2] + dz]
        if dx or dy or dz:
            self._goto(target, gripper_open=self._gripper_open, steps=self.servo_steps)
        if grip == "close" and self._gripper_open:
            self._gripper_open = False
            hold = self._tcp_pos() or target
            self._goto(hold, gripper_open=False, steps=self.settle_steps, tol=0.0)
            gw = self._gripper_width()
            self._holding = bool(gw > self.grip_hold_eps)
            print(f"[isaac_env] servo: close gripper_width={gw:.4f} -> "
                  f"holding={self._holding}")
        elif grip == "open" and not self._gripper_open:
            self._gripper_open = True
            hold = self._tcp_pos() or target
            self._goto(hold, gripper_open=True, steps=self.settle_steps, tol=0.0)
            self._holding = False
        return True

    def _place(self, xyz: list) -> bool:
        above = [xyz[0], xyz[1], xyz[2] + self.approach_z + 0.04]
        drop = [xyz[0], xyz[1], xyz[2] + self.grasp_z + 0.04]
        self._goto(above, gripper_open=False)
        self._goto(drop, gripper_open=False)
        self._goto(drop, gripper_open=True, steps=self.settle_steps, tol=0.0)  # release
        self._goto(above, gripper_open=True)
        self._holding = False
        self._gripper_open = True
        return True

    # ------------------------------------------------------------- sensing
    def _gripper_width(self) -> float:
        robot = self._scene()["robot"]
        ids = [i for i, n in enumerate(robot.data.joint_names) if "finger" in n]
        return float(robot.data.joint_pos[0, ids].sum()) if ids else 0.0

    def _grasp_metrics(self) -> dict:
        """Grasp state from proprioception only — whether a grasp is currently held
        (determined when the servo loop closes the gripper, from gripper width) plus
        the live gripper width. No ground-truth object position is read."""
        return {"grasped": bool(self._holding),
                "gripper_width": round(self._gripper_width(), 4),
                "object_height": None}

    def metrics(self) -> dict:
        grasp = self._grasp_metrics()
        grasp["objects"] = self.objects_world()
        grasp["holding"] = bool(self._holding)
        return grasp

    def _report(self, command: str, ok: bool, error: str = "") -> dict:
        m = self.metrics()
        return {"command": command, "ok": ok, "error": error,
                "holding": bool(self._holding), "objects": m["objects"],
                "grasp": {k: m[k] for k in ("grasped", "gripper_width", "object_height")}}

    # ------------------------------------------------------- capabilities
    def capabilities(self) -> dict:
        am = self.env.unwrapped.action_manager
        terms = []
        ee_body = None
        try:
            names = list(am.active_terms)
            dims = list(am.action_term_dim)
            for name, dim in zip(names, dims):
                term = am.get_term(name) if hasattr(am, "get_term") else None
                terms.append({"name": name, "dim": int(dim),
                              "type": type(term).__name__ if term else "?"})
                # The controlled end-effector body is carried on the IK action term's
                # cfg — read it instead of hardcoding a robot-specific body name.
                cfg = getattr(term, "cfg", None)
                body = getattr(cfg, "body_name", None) if cfg is not None else None
                if body:
                    ee_body = body
        except Exception:
            # Don't silently swallow — a broken introspection should be visible, not
            # masquerade as an empty action space.
            print(f"[isaac_env] capabilities: action-space introspection failed:\n"
                  f"{traceback.format_exc()}")
        # Derive the action-space type from the live terms rather than hardcoding it,
        # so this stays accurate for a different arm/action configuration.
        space_type = " + ".join(t["type"] for t in terms) if terms else "?"
        robot = self._scene()["robot"]
        return {
            "action_space": {
                "type": space_type,
                "dof": int(getattr(am, "total_action_dim", 0)),
                "terms": terms,
                "ee_body": ee_body or "?",
                "arm_joints": [n for n in robot.data.joint_names if "finger" not in n],
                "gripper": {"open": P.GRIPPER_OPEN, "close": P.GRIPPER_CLOSE,
                            "joints": [n for n in robot.data.joint_names if "finger" in n]},
                "pose_format": "[px,py,pz,qw,qx,qy,qz] in robot root frame",
            },
            "skills": SKILLS,
            "objects": self.objects_world(),
            "obstacles": self.obstacles,
            "camera": {"width": self.CAM_W, "height": self.CAM_H,
                       "name": self.camera_name},
            "control": {"pos_tol": self.pos_tol, "servo_max": self.servo_max},
        }

    # ------------------------------------------------------------- camera
    def frame(self) -> bytes:
        import io

        from PIL import Image

        rgb = self._scene()[self.camera_name].data.output["rgb"][0]
        arr = rgb.detach().cpu().numpy()[..., :3].astype("uint8")
        self.CAM_H, self.CAM_W = int(arr.shape[0]), int(arr.shape[1])
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    def frame_rgb(self) -> bytes:
        """The latest camera frame as raw RGB24 bytes (H*W*3) — what the RTSP
        push streamer feeds to its encoder (no PNG round-trip per frame)."""
        rgb = self._scene()[self.camera_name].data.output["rgb"][0]
        arr = rgb.detach().cpu().numpy()[..., :3].astype("uint8")
        return arr.tobytes()

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
