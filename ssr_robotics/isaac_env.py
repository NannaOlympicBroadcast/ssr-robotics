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
top of those primitives the bridge implements a few skills (pick / place_on /
place_at / move_above / raw) the agent may invoke; the agent discovers them via
``arm_describe`` and plans accordingly.

The Isaac Sim app MUST be launched (``isaaclab.app.AppLauncher``) before
constructing this class — see ``run_bridge.py`` / ``openarm_isaac_lab/scripts``.
"""

from __future__ import annotations

from . import protocol as P

# Maps scene rigid-object keys to friendly names the agent sees. Mirrors
# OBJECT_FRIENDLY_NAMES in the openarm manip env cfg.
OBJECT_FRIENDLY = {"object": "apple", "orange": "orange"}

# Skills the bridge implements on top of the arm's primitive action types.
SKILLS = [
    {"name": "pick", "args": {"object": "str"}, "grasping": True,
     "desc": "grasp a named object and lift it clear of the surface"},
    {"name": "place_on", "args": {"object": "str"},
     "desc": "place the currently-held object on top of another named object"},
    {"name": "place_at", "args": {"x": "float", "y": "float"},
     "desc": "place the currently-held object at a world (x, y) location"},
    {"name": "move_above", "args": {"object": "str?", "x": "float?", "y": "float?"},
     "desc": "move the end-effector above an object or an (x, y) location"},
    {"name": "raw", "args": {"actions": "list[float vectors]"},
     "desc": "replay low-level action vectors matching action_space.dof"},
]

# Fixed top-down grasp orientation (quaternion wxyz) in the robot root frame.
# Tune on the real robot if the gripper approach differs.
GRASP_QUAT = (0.0, 1.0, 0.0, 0.0)

APPROACH_Z = 0.12   # hover height above an object before descending (m)
GRASP_Z = 0.02      # TCP height above the object centre when closing (m)
LIFT_Z = 0.25       # height to lift to after grasping (m)


class IsaacOpenArmEnv:
    def __init__(self, task: str = "Isaac-Manip-OpenArm-v0", steps_per_move: int = 30,
                 settle_steps: int = 15, num_envs: int = 1):
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
        # The `env_cfg_entry_point` gym.make() is registered with is inert metadata —
        # Isaac Lab requires resolving it into a real cfg instance first. This also
        # forces num_envs=1: the task's registered default is RL-training scale
        # (e.g. 4096), but this bridge drives a single real/simulated arm.
        env_cfg = parse_env_cfg(task, num_envs=num_envs)
        self.env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
        self.device = self.env.unwrapped.device
        cam = self.env.unwrapped.scene["tiled_camera"]
        self.CAM_H, self.CAM_W = int(cam.image_shape[0]), int(cam.image_shape[1])
        self._holding: str | None = None
        self.reset()

    # ------------------------------------------------------------------ env
    def reset(self, target: str | None = None) -> None:
        self.env.reset()
        self._holding = None

    def _scene(self):
        return self.env.unwrapped.scene

    @property
    def _num(self) -> int:
        return self.env.unwrapped.num_envs

    # ------------------------------------------------------- object lookup
    def _object_key(self, friendly: str) -> str | None:
        for key, name in OBJECT_FRIENDLY.items():
            if name == friendly and key in self._scene().rigid_objects:
                return key
        return friendly if friendly in self._scene().rigid_objects else None

    def _object_pos_root(self, key: str):
        """Object position in the robot root frame (x, y, z)."""
        scene = self._scene()
        robot = scene["robot"]
        obj_w = scene[key].data.root_pos_w[:, :3]
        pos_b, _ = self._sub(robot.data.root_pos_w, robot.data.root_quat_w, obj_w)
        return pos_b[0]

    def objects_world(self) -> dict:
        scene = self._scene()
        out = {}
        for key in scene.rigid_objects.keys():
            name = OBJECT_FRIENDLY.get(key, key)
            p = scene[key].data.root_pos_w[0]
            out[name] = [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)]
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
        try:
            ok = self._run_skill(req)
            return self._report(req.command, ok=ok)
        except Exception as e:  # report failures rather than crashing the bridge
            return self._report(req.command, ok=False, error=str(e))

    def _run_skill(self, req: "P.ArmActionRequest") -> bool:
        cmd, args = req.command, req.args
        if cmd == "raw":
            t = self.torch
            for vec in req.actions:
                a = t.tensor([vec] * self._num, dtype=t.float32, device=self.device)
                self.env.step(a)
            return True
        if cmd == "move_above":
            pos = self._target_xyz(args)
            pos[2] = pos[2] + APPROACH_Z
            self._goto(pos, gripper_open=self._holding is None)
            return True
        if cmd == "pick":
            return self._pick(str(args.get("object", "")))
        if cmd == "place_on":
            return self._place(self._target_xyz({"object": args.get("object", "")}))
        if cmd == "place_at":
            return self._place(self._target_xyz(args))
        raise ValueError(f"unknown skill '{cmd}'")

    def _target_xyz(self, args: dict):
        """Resolve a target to a robot-root-frame [x, y, z] list (mutable)."""
        if args.get("object"):
            key = self._object_key(str(args["object"]))
            if key is None:
                raise ValueError(f"object '{args['object']}' not in scene")
            p = self._object_pos_root(key)
            return [float(p[0]), float(p[1]), float(p[2])]
        if "x" in args and "y" in args:
            # z defaults to table height (approx object resting height).
            return [float(args["x"]), float(args["y"]), 0.055]
        raise ValueError("target needs an 'object' or 'x'+'y'")

    def _pick(self, friendly: str) -> bool:
        key = self._object_key(friendly)
        if key is None:
            raise ValueError(f"object '{friendly}' not in scene")
        p = self._object_pos_root(key)
        above = [float(p[0]), float(p[1]), float(p[2]) + APPROACH_Z]
        grasp = [float(p[0]), float(p[1]), float(p[2]) + GRASP_Z]
        self._goto(above, gripper_open=True)
        self._goto(grasp, gripper_open=True)
        self._goto(grasp, gripper_open=False, steps=self.settle_steps)  # close
        lift = [float(p[0]), float(p[1]), float(p[2]) + LIFT_Z]
        self._goto(lift, gripper_open=False)
        ok = self._grasp_metrics(key)["grasped"]
        self._holding = friendly if ok else None
        return ok

    def _place(self, xyz: list) -> bool:
        above = [xyz[0], xyz[1], xyz[2] + APPROACH_Z + 0.04]
        drop = [xyz[0], xyz[1], xyz[2] + GRASP_Z + 0.04]
        self._goto(above, gripper_open=False)
        self._goto(drop, gripper_open=False)
        self._goto(drop, gripper_open=True, steps=self.settle_steps)  # release
        self._goto(above, gripper_open=True)
        self._holding = None
        return True

    # ------------------------------------------------------------- sensing
    def _gripper_width(self) -> float:
        robot = self._scene()["robot"]
        ids = [i for i, n in enumerate(robot.data.joint_names) if "finger" in n]
        return float(robot.data.joint_pos[0, ids].sum()) if ids else 0.0

    def _grasp_metrics(self, key: str | None = None) -> dict:
        scene = self._scene()
        gripper_width = self._gripper_width()
        grasped = False
        height = None
        if key and key in scene.rigid_objects:
            obj_w = scene[key].data.root_pos_w[0]
            ee_w = scene["ee_frame"].data.target_pos_w[0, 0]
            dist = float(self.torch.linalg.norm(obj_w - ee_w))
            height = float(obj_w[2])
            grasped = bool(height > 0.10 and dist < 0.10 and gripper_width < 0.03)
        return {"grasped": grasped, "gripper_width": round(gripper_width, 4),
                "object_height": round(height, 4) if height is not None else None}

    def metrics(self) -> dict:
        grasp = self._grasp_metrics(self._object_key(self._holding) if self._holding else None)
        grasp["objects"] = self.objects_world()
        grasp["holding"] = self._holding
        return grasp

    def _report(self, command: str, ok: bool, error: str = "") -> dict:
        m = self.metrics()
        return {"command": command, "ok": ok, "error": error,
                "holding": self._holding, "objects": m["objects"],
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
