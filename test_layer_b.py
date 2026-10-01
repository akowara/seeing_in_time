"""
Tests for Layer B step 4 (trap avoider) and the full LayerB loop.

Run:  python test_layer_b.py      (or: pytest test_layer_b.py)
Runs on CPU; needs heading_memory.py, trap_detector.py, escape_controller.py, layer_b.py.
"""

import math
import torch

from heading_memory import HeadingMemory
from layer_b import TrapAvoider, LayerB

DEV = 'cpu'
DT = 1 / 15


def at(x=0.0, y=0.0, z=1.0):
    return torch.tensor([[x, y, z]], dtype=torch.float32)


def rays(yaws_deg):
    out = [[math.cos(math.radians(y)), math.sin(math.radians(y)), 0.0] for y in yaws_deg]
    return torch.tensor([out], dtype=torch.float32)


def dists(values):
    return torch.tensor([values], dtype=torch.float32)


def heading_deg(v):
    return math.degrees(math.atan2(v[1].item(), v[0].item()))


GOAL = at(30.0)
GOAL_V = torch.tensor([[5.0, 0.0, 0.0]])


def active_avoider(trap_x=10.0, pos=None, radius=6.0, memory=None, goal=GOAL):
    av = TrapAvoider(1, DEV, radius=radius)
    av.activate(torch.tensor([True]), 0.0, at(trap_x), pos if pos is not None else at(), goal, memory)
    return av


# =============================================================================
# TrapAvoider unit tests
# =============================================================================

def test_inactive_passes_through():
    av = TrapAvoider(1, DEV)
    v, active = av.step(0.0, at(), GOAL, GOAL_V)
    assert not active[0] and torch.allclose(v, GOAL_V)


def test_goal_inside_circle_turned_to_tangent():
    av = active_avoider(trap_x=10.0, radius=6.0)                # trap 10 m ahead, goal behind it
    v, active = av.step(0.0, at(), GOAL, GOAL_V)
    expected = math.degrees(math.asin(6.0 / 10.0))              # tangent half-angle, ~36.9 deg
    assert active[0]
    assert abs(abs(heading_deg(v[0])) - expected) < 1e-3, heading_deg(v[0])
    assert abs(v[0, :2].norm().item() - 5.0) < 1e-5             # speed kept


def test_goal_missing_circle_passes_through():
    av = active_avoider(trap_x=10.0, radius=6.0)
    gv = torch.tensor([[0.0, 5.0, 0.0]])                        # heading 90 deg: misses the circle
    v, _ = av.step(0.0, at(), GOAL, gv)
    assert torch.allclose(v, gv)


def test_inside_circle_flies_tangent():
    av = active_avoider(trap_x=10.0, pos=at(), radius=6.0)
    v, _ = av.step(0.0, at(x=7.0), GOAL, GOAL_V)                # 3 m from the trap: inside
    assert abs(abs(heading_deg(v[0])) - 90.0) < 1e-3, heading_deg(v[0])


def test_side_picked_by_memory():
    mem = HeadingMemory(1, DEV)
    # +37 deg (left of the trap) blocked, -37 deg (right) open
    mem.update(0.0, at(), rays([37, -37]), dists([1.0, 20.0]))
    av = active_avoider(trap_x=10.0, radius=6.0, memory=mem)
    v, _ = av.step(0.0, at(), GOAL, GOAL_V)
    assert heading_deg(v[0]) < 0, heading_deg(v[0])            # goes right


def test_side_picked_toward_goal_when_memory_silent():
    goal = at(30.0, -10.0)                                      # goal off to the right
    av = active_avoider(trap_x=10.0, radius=6.0, goal=goal)
    gv = torch.tensor([[5.0, -0.5, 0.0]])                       # still pointing into the circle
    v, _ = av.step(0.0, at(), goal, gv)
    assert heading_deg(v[0]) < 0, heading_deg(v[0])


def test_side_does_not_flip():
    av = active_avoider(trap_x=10.0, radius=6.0)
    v0, _ = av.step(0.0, at(), GOAL, GOAL_V)
    s0 = math.copysign(1, heading_deg(v0[0]))
    gv = torch.tensor([[5.0, -1.0, 0.0]])                      # goal direction now slightly the other way
    v1, _ = av.step(0.1, at(0.5, 0.0), GOAL, gv)
    assert math.copysign(1, heading_deg(v1[0])) == s0


def test_released_once_past_trap():
    av = active_avoider(trap_x=10.0, radius=6.0)               # trap 20 m from the goal
    _, active = av.step(1.0, at(x=12.0, y=7.0), GOAL, GOAL_V)  # ~19.3 m from goal: not yet
    assert active[0]
    _, active = av.step(2.0, at(x=19.0, y=5.0), GOAL, GOAL_V)  # ~12.1 m < 20 - 2: past it
    assert not active[0]


def test_released_on_timeout():
    av = TrapAvoider(1, DEV, timeout=30.0)
    av.activate(torch.tensor([True]), 0.0, at(10.0), at(), GOAL)
    _, active = av.step(29.9, at(), GOAL, GOAL_V)
    assert active[0]
    _, active = av.step(30.0, at(), GOAL, GOAL_V)
    assert not active[0]


# =============================================================================
# Full Layer B, closed loop in the U-trap
# =============================================================================

WALLS = [(12, -3, 12, 3), (8, 3, 12, 3), (8, -3, 12, -3)]
CAM = [-39 + 78 * i / 16 for i in range(17)]


def cast(ox, oy, yaw_rad, walls=None):
    walls = WALLS if walls is None else walls       # read the global at call time
    dx, dy = math.cos(yaw_rad), math.sin(yaw_rad)
    best = float('inf')
    for (ax, ay, bx, by) in walls:
        ex, ey = bx - ax, by - ay
        rx, ry = ax - ox, ay - oy
        det = -dx * ey + ex * dy
        if abs(det) < 1e-12:
            continue
        s = (-rx * ey + ex * ry) / det
        u = (dx * ry - dy * rx) / det
        if s > 1e-9 and 0.0 <= u <= 1.0:
            best = min(best, s)
    return best


def inside_cup(x, y):
    return 8.0 < x < 12.0 and abs(y) < 3.0


def run(seconds=25.0, goal_xy=(20.0, 0.0), T=0.5, radius=0.5, v_max=5.0, progress='goal'):
    """Point mass with lag T that stops at walls; camera faces the desired direction."""
    goal = at(*goal_xy)
    lb = LayerB(1, DEV, detector_kw=dict(progress=progress))
    p, v, head = [0.0, 0.0], [0.0, 0.0], 0.0
    log = []
    for k in range(int(seconds / DT)):
        t = k * DT
        pos = torch.tensor([[p[0], p[1], 1.0]])
        g = (goal - pos)[0].clone()
        g[2] = 0.0
        goal_vel = (g / g.norm() * min(g.norm().item(), v_max))[None]

        d = [cast(p[0], p[1], head + math.radians(a)) for a in CAM]
        des, info = lb.step(t, pos, goal, goal_vel,
                            rays([math.degrees(head) + a for a in CAM]), dists(d))
        dx, dy = des[0, 0].item(), des[0, 1].item()
        if dx * dx + dy * dy > 1e-9:
            head = math.atan2(dy, dx)

        v[0] += (dx - v[0]) * DT / T
        v[1] += (dy - v[1]) * DT / T
        sp = math.hypot(v[0], v[1])
        if sp > 1e-6:
            if cast(p[0], p[1], math.atan2(v[1], v[0])) < sp * DT + radius:
                v = [0.0, 0.0]
            else:
                p = [p[0] + v[0] * DT, p[1] + v[1] * DT]
        log.append((t, p[0], p[1], bool(info['escaping'][0]), bool(info['avoiding'][0])))
    return log, lb


def test_full_layer_b_reaches_goal_behind_utrap():
    log, lb = run()
    reached = [t for (t, x, y, _, _) in log if math.hypot(20.0 - x, y) < 1.5]
    assert reached, f"never reached the goal; ended at {log[-1][1]:.1f}, {log[-1][2]:.1f}"
    assert lb.escape.n_escapes[0].item() == 1, lb.escape.n_escapes[0].item()   # one escape was enough


def test_command_progress_mode_also_reaches_goal():
    log, lb = run(progress='command')
    assert any(math.hypot(20.0 - x, y) < 1.5 for (_, x, y, _, _) in log)
    assert lb.escape.n_escapes[0].item() == 1
    assert lb.n_retrap_avoiding[0].item() == 0


def test_slow_drone_detour_not_mistaken_for_trap():
    """A slow, sluggish drone (2 m/s, 1 s lag) going around the cup: with goal-progress
    it can look stuck mid-detour; with command-progress it shouldn't be re-trapped."""
    _, lb = run(seconds=60.0, v_max=2.0, T=1.0, progress='command')
    assert lb.n_retrap_avoiding[0].item() == 0, lb.n_retrap_avoiding[0].item()


def test_never_reenters_cup_after_escape():
    log, _ = run()
    exit_idx = next(i for i in range(1, len(log)) if log[i - 1][3] and not log[i][3])
    assert not any(inside_cup(x, y) for (_, x, y, _, _) in log[exit_idx:])


def test_avoidance_used_then_released():
    log, lb = run()
    assert any(a for (_, _, _, _, a) in log)                   # avoided the trap at some point
    assert not lb.avoider.active[0]                            # and let go once past it


def test_open_scene_layer_b_stays_out_of_the_way():
    """No walls: Layer B should never trigger and the drone flies straight to the goal."""
    global WALLS
    saved, WALLS = WALLS, []
    try:
        log, lb = run(seconds=8.0)
    finally:
        WALLS = saved
    assert lb.escape.n_escapes[0].item() == 0
    assert not any(e or a for (_, _, _, e, a) in log)
    assert any(math.hypot(20.0 - x, y) < 1.5 for (_, x, y, _, _) in log)


# =============================================================================

if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}  {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")