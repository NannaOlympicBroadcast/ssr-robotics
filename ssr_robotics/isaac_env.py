"""Adapter for the real Isaac Lab OpenArm manipulation environment.

Wraps ``Isaac-Manip-OpenArm-v0`` (the IK + camera + apple/orange variant added in
``openarm_isaac_lab``) behind the env interface :class:`EnvRunner` expects:
``reset()``, ``execute(req)``, ``metrics()``, ``frame()``, ``capabilities()`` and
``CAM_W`` / ``CAM_H``.

**Capabilities are introspected from the live env**, not hardcoded: the action
space is read from ``env.action_manager`` (term names, dims, classes) and the
objects from ``env.scene.rigid_objects``. The arm's real action types — grounded
in the OpenArm repo — are end-effector pose (``DifferentialInverseKinematicsAction``
on body ``openarm_hand``) plus a binary gripper (``openarm_finger_joint.*``). On
top of those primitives the bridge implements a few **object-agnostic** skills
(pick / place_at / move_above / raw) the agent may invoke; the agent discovers
them via ``arm_describe`` and plans accordingly. The bridge knows nothing about
*what* it manipulates — every skill is driven by world coordinates the agent
supplies from its own perception of the camera frame, never by a hardcoded object
name or identity — so it generalizes beyond any particular scene.

The Isaac Sim app MUST be launched (``isaaclab.app.AppLauncher``) before
constructing this class — see ``run_bridge.py`` / ``openarm_isaac_lab/scripts``.
"""

from __future__ import annotations

import math
import threading

from . import protocol as P
from . import vision as V

# Skills the bridge implements on top of the arm's primitive action types. They are
# deliberately object-agnostic — the bridge knows nothing about *what* it is
# manipulating, so this generalizes beyond any particular scene. Coordinates are in
# the robot ROOT frame; the agent gets candidate object positions from the vision
# snapshot (arm_describe / arm_get_scene "objects", detected by camera) and never
# refers to objects by a hardcoded name/identity.
SKILLS = [
    {"name": "pick", "args": {"x": "float", "y": "float", "z": "float?"},
     "grasping": True,
     "desc": "grasp the object nearest root-frame (x, y) and lift it; the robot "
             "pinpoints the exact grasp point by vision (camera), so (x, y) need "
             "only be approximate"},
    {"name": "place_at", "args": {"x": "float", "y": "float"},
     "desc": "place the currently-held object at root-frame (x, y)"},
    {"name": "move_above", "args": {"x": "float", "y": "float"},
     "desc": "move the end-effector above root-frame (x, y)"},
    {"name": "raw", "args": {"actions": "list[float vectors]"},
     "desc": "replay low-level action vectors matching action_space.dof"},
]

# Fixed top-down grasp orientation (quaternion wxyz) in the robot root frame.
# Tune on the real robot if the gripper approach differs.
GRASP_QUAT = (0.0, 1.0, 0.0, 0.0)

APPROACH_Z = 0.12   # hover height above a target before descending (m)
# TCP height relative to the target point when closing (m).
GRASP_Z = 0.0
LIFT_Z = 0.25       # height to lift to after grasping (m)

# World height (m) of the plane the camera back-projects onto when locating an
# object by vision — the approximate object resting height on the table — and the
# default target z for a pick/place when only x, y are given.
TABLE_Z = 0.055

# Summed finger-joint width above which the gripper is judged to be holding
# something after a close (an object wedged the fingers open). Proprioceptive, so
# grasp success needs no ground-truth object state. Tune on the real robot.
GRIP_HOLD_EPS = 0.005


class IsaacOpenArmEnv:
    def __init__(self, task: str = "Isaac-Manip-OpenArm-v0", steps_per_move: int = 30,
                 settle_steps: int = 15, steps_per_waypoint: int = 10, num_envs: int = 1):
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
        # The `env_cfg_entry_point` gym.make() is registered with is inert metadata —
        # Isaac Lab requires resolving it into a real cfg instance first. This also
        # forces num_envs=1: the task's registered default is RL-training scale
        # (e.g. 4096), but this bridge drives a single real/simulated arm.
        env_cfg = parse_env_cfg(task, num_envs=num_envs)
        self.env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
        self.device = self.env.unwrapped.device
        cam = self.env.unwrapped.scene["tiled_camera"]
        self.CAM_H, self.CAM_W = int(cam.image_shape[0]), int(cam.image_shape[1])
        self._holding: bool = False  # whether a grasp is currently held
        # Set from another thread (the bus-callback handler on reset) to ask a
        # running multi-step skill to bail out, so a reset doesn't have to wait for
        # a long raw replay to finish before it can run.
        self._interrupt = threading.Event()
        self.reset()

    def interrupt(self) -> None:
        """Ask an in-flight multi-step skill (e.g. a raw replay) to stop ASAP.

        Checked between waypoints, so it cannot break out of a single hung
        ``env.step``, but it lets a reset preempt a long/runaway replay."""
        self._interrupt.set()

    # ------------------------------------------------------------------ env
    def reset(self, target: str | None = None) -> None:
        self._interrupt.clear()
        self.env.reset()
        self._holding = False

    def _scene(self):
        return self.env.unwrapped.scene

    @property
    def _num(self) -> int:
        return self.env.unwrapped.num_envs

    # ------------------------------------------------------------ perception
    def _perceive(self) -> list[dict]:
        """Detect graspable objects BY VISION from the overhead camera.

        Returns a list (largest blob first) of detections, each with the centroid
        ``pixel``, blob ``pixels`` count, and the back-projected position in both
        ``world`` and robot-``root`` frames. Uses NO ground-truth scene state and
        no hardcoded object identity — see :mod:`ssr_robotics.vision`.
        """
        from isaaclab.utils.math import matrix_from_quat

        cam = self._scene()["tiled_camera"]
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype("float32")
        blobs = V.connected_blobs(V.foreground_mask(rgb))
        K = cam.data.intrinsic_matrices[0].detach().cpu().numpy()
        cam_pos = cam.data.pos_w[0].detach().cpu().numpy()
        cam_rot = matrix_from_quat(cam.data.quat_w_ros[0]).detach().cpu().numpy()
        t = self.torch
        robot = self._scene()["robot"]
        dets = []
        for (u, v, n) in blobs:
            pw = V.pixel_to_table_point(u, v, K, cam_pos, cam_rot, TABLE_Z)
            if pw is None:
                continue
            pwt = t.tensor([pw], dtype=t.float32, device=self.device)
            pb, _ = self._sub(robot.data.root_pos_w, robot.data.root_quat_w, pwt)
            dets.append({
                "pixel": [round(float(u), 1), round(float(v), 1)], "pixels": int(n),
                "world": [round(float(c), 4) for c in pw],
                "root": [round(float(pb[0, 0]), 4), round(float(pb[0, 1]), 4),
                         round(float(pb[0, 2]), 4)],
            })
        return dets

    def objects_world(self) -> dict:
        """Perceived object positions (robot ROOT frame) BY VISION, keyed generically
        (obj0, obj1, … largest blob first). No ground-truth scene state — this is the
        situational awareness the agent uses to choose a pick coordinate."""
        dets = self._perceive()
        out = {f"obj{i}": d["root"] for i, d in enumerate(dets)}
        print(f"[isaac_env] vision: detected {len(dets)} object(s) -> {out}")
        return out

    # --------------------------------------------------------- IK stepping
    def _ik_action(self, pos, gripper_open: bool):
        """Build an [px,py,pz,qw,qx,qy,qz, gripper] action for all envs."""
        t = self.torch
        g = P.GRIPPER_OPEN if gripper_open else P.GRIPPER_CLOSE
        vec = [float(pos[0]), float(pos[1]), float(pos[2]), *GRASP_QUAT, g]
        return t.tensor([vec] * self._num, dtype=t.float32, device=self.device)

    def _goto(self, pos, gripper_open: bool, steps: int | None = None) -> None:
        action = self._ik_action(pos, gripper_open)
        for _ in range(steps or self.steps_per_move):
            self.env.step(action)

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
                    self.env.step(a)
            print(f"[isaac_env] raw: done ({len(actions)} waypoints)")
            return True
        if cmd == "move_above":
            pos = self._target_xyz(args)
            pos[2] = pos[2] + APPROACH_Z
            self._goto(pos, gripper_open=not self._holding)
            return True
        if cmd == "pick":
            return self._pick(self._target_xyz(args))
        if cmd == "place_at":
            return self._place(self._target_xyz(args))
        raise ValueError(f"unknown skill '{cmd}'")

    def _target_xyz(self, args: dict):
        """Resolve a coordinate target to a robot-root-frame [x, y, z] list.

        Object-agnostic: the agent supplies the coordinates (from its own
        perception of the camera frame), the bridge just executes them."""
        if "x" in args and "y" in args:
            z = float(args["z"]) if args.get("z") is not None else TABLE_Z
            return [float(args["x"]), float(args["y"]), z]
        raise ValueError("target needs 'x' and 'y' (and optionally 'z')")

    def _grasp_point(self, xyz: list) -> list:
        """Refine an approximate target to the nearest VISION-detected object (robot
        root frame): the robot decides the exact grasp position by looking, not from
        ground-truth state. Raises if vision sees nothing — never falls back to
        privileged scene data."""
        dets = self._perceive()
        if not dets:
            raise ValueError("vision: no graspable object detected in the camera frame")
        tx, ty = float(xyz[0]), float(xyz[1])
        best = min(dets, key=lambda d: (d["root"][0] - tx) ** 2 + (d["root"][1] - ty) ** 2)
        print(f"[isaac_env] vision: pick hint ({tx:.4f},{ty:.4f}) -> nearest detection "
              f"root={best['root']} pixel={best['pixel']} pixels={best['pixels']}")
        return list(best["root"])

    def _pick(self, xyz: list) -> bool:
        p = self._grasp_point(xyz)  # exact grasp point comes from vision
        above = [p[0], p[1], p[2] + APPROACH_Z]
        grasp = [p[0], p[1], p[2] + GRASP_Z]
        self._goto(above, gripper_open=True)
        self._goto(grasp, gripper_open=True)
        self._goto(grasp, gripper_open=False, steps=self.settle_steps)  # close
        lift = [p[0], p[1], p[2] + LIFT_Z]
        self._goto(lift, gripper_open=False)
        # Proprioceptive success: after closing, an object wedges the fingers open.
        gw = self._gripper_width()
        ok = bool(gw > GRIP_HOLD_EPS)
        print(f"[isaac_env] pick: gripper_width={gw:.4f} -> grasped={ok}")
        self._holding = ok
        return ok

    def _place(self, xyz: list) -> bool:
        above = [xyz[0], xyz[1], xyz[2] + APPROACH_Z + 0.04]
        drop = [xyz[0], xyz[1], xyz[2] + GRASP_Z + 0.04]
        self._goto(above, gripper_open=False)
        self._goto(drop, gripper_open=False)
        self._goto(drop, gripper_open=True, steps=self.settle_steps)  # release
        self._goto(above, gripper_open=True)
        self._holding = False
        return True

    # ------------------------------------------------------------- sensing
    def _gripper_width(self) -> float:
        robot = self._scene()["robot"]
        ids = [i for i, n in enumerate(robot.data.joint_names) if "finger" in n]
        return float(robot.data.joint_pos[0, ids].sum()) if ids else 0.0

    def _grasp_metrics(self) -> dict:
        """Grasp state from proprioception only — whether a grasp is currently held
        (determined at pick time from gripper width) plus the live gripper width. No
        ground-truth object position is read."""
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
        try:
            names = list(am.active_terms)
            dims = list(am.action_term_dim)
            for name, dim in zip(names, dims):
                term = am.get_term(name) if hasattr(am, "get_term") else None
                terms.append({"name": name, "dim": int(dim),
                              "type": type(term).__name__ if term else "?"})
        except Exception:
            pass
        robot = self._scene()["robot"]
        return {
            "action_space": {
                "type": "DifferentialInverseKinematicsAction(pose)+BinaryGripper",
                "dof": int(getattr(am, "total_action_dim", 0)),
                "terms": terms,
                "ee_body": "openarm_hand",
                "arm_joints": [n for n in robot.data.joint_names if "finger" not in n],
                "gripper": {"open": P.GRIPPER_OPEN, "close": P.GRIPPER_CLOSE,
                            "joints": [n for n in robot.data.joint_names if "finger" in n]},
                "pose_format": "[px,py,pz,qw,qx,qy,qz] in robot root frame",
            },
            "skills": SKILLS,
            "objects": self.objects_world(),
            "camera": {"width": self.CAM_W, "height": self.CAM_H},
        }

    # ------------------------------------------------------------- camera
    def frame(self) -> bytes:
        import io

        from PIL import Image

        rgb = self._scene()["tiled_camera"].data.output["rgb"][0]
        arr = rgb.detach().cpu().numpy()[..., :3].astype("uint8")
        self.CAM_H, self.CAM_W = int(arr.shape[0]), int(arr.shape[1])
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
