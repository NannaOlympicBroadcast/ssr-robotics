"""Unit tests for the pure (no-GPU) helpers added to the Isaac env adapter.

The full :class:`~ssr_robotics.isaac_env.IsaacOpenArmEnv` only runs on the robot
machine (it imports isaaclab/torch in ``__init__``), but the module-level helpers
it relies on — waypoint-arrival tolerance and obstacle parsing — are pure Python
and back the four grasp-reliability fixes (tolerant waypoints, advertised collision
obstacles), so they are pinned down here without a simulator.
"""

from __future__ import annotations

from ssr_robotics import isaac_env as E


# ----------------------------------------------------------- waypoint tolerance
def test_within_tol_true_when_inside_radius():
    # 3-4-5: distance 0.05 from target, tolerance 0.06 → reached.
    assert E._within_tol([0.0, 0.0, 0.0], [0.03, 0.04, 0.0], 0.06) is True


def test_within_tol_false_when_outside_radius():
    assert E._within_tol([0.0, 0.0, 0.0], [0.03, 0.04, 0.0], 0.04) is False


def test_within_tol_exact_boundary_counts_as_reached():
    # Euclidean distance exactly == tol is "reached" (<=, not <).
    assert E._within_tol([0.0, 0.0, 0.0], [0.01, 0.0, 0.0], 0.01) is True


def test_within_tol_none_current_is_never_reached():
    # TCP unreadable → fall back to running the fixed step budget, never early-exit.
    assert E._within_tol(None, [0.0, 0.0, 0.0], 0.01) is False


def test_within_tol_nonpositive_tol_disables_early_exit():
    assert E._within_tol([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0) is False
    assert E._within_tol([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], -1.0) is False


# --------------------------------------------------------------- obstacle parse
def test_parse_obstacles_reads_named_aabbs():
    raw = '[{"name": "stand", "aabb": [0, 0, 0, 0.1, 0.1, 0.2]}]'
    obs = E._parse_obstacles(raw)
    assert obs == [{"name": "stand", "aabb": [0.0, 0.0, 0.0, 0.1, 0.1, 0.2]}]


def test_parse_obstacles_names_anonymous_entries():
    obs = E._parse_obstacles('[{"aabb": [0, 0, 0, 1, 1, 1]}]')
    assert obs == [{"name": "obstacle0", "aabb": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]}]


def test_parse_obstacles_drops_malformed_entries():
    # wrong-length aabb, non-numeric aabb, non-dict, missing aabb — all dropped.
    raw = ('[{"aabb": [1, 2, 3]}, {"aabb": ["x", 0, 0, 0, 0, 0]}, '
           '"nope", {"name": "no_aabb"}]')
    assert E._parse_obstacles(raw) == []


def test_parse_obstacles_blank_and_garbage_return_empty():
    for raw in ("", "   ", None, "not json", "{}", '{"aabb": [0,0,0,1,1,1]}'):
        assert E._parse_obstacles(raw) == []
