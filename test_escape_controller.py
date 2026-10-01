"""
Tests for Layer B step 3 (escape controller).

Run:  python test_escape_controller.py      (or: pytest test_escape_controller.py)
Runs on CPU; needs heading_memory.py, trap_detector.py and escape_controller.py.
"""

import math
import torch

from heading_memory import HeadingMemory
from trap_detector import TrapDetector
from escape_controller import EscapeController

DEV = 'cpu'
DT = 1 / 15


# =============================================================================
# Helpers
# =============================================================================

def rays(yaws_deg):
    out = [[math.cos(math.radians(y)), math.sin(math.radians(y)), 0.0] for y in yaws_deg]
    return torch.tensor([out], dtype=torch.float32)


def dists(values):
    return torch.tensor([values], dtype=torch.float32)


def at(x=0.0, y=0.0, z=1.0):
    return torch.tensor([[x, y, z]], dtype=torch.float32)


def heading_deg(v):
    return math.degrees(math.atan2(v[1].item(), v[0].item()))


class FakeDetector:
    """Stands in for TrapDetector in unit tests: trapped state set by hand."""
    def __init__(self, trapped, trap_yaw_deg=0.0):
        B = len(trapped)
        self.trapped = torch.tensor(trapped)
        self.trap_yaw = torch.full((B,), math.radians(trap_yaw_deg))
        self.trap_pos = torch.zeros(B, 3)
        self.cleared = torch.zeros(B, dtype=torch.bool)

    def clear(self, mask, t):
        self.trapped &= ~mask
        self.cleared |= mask


GOAL_V = torch.tensor([[5.0, 0.0, 0.0]])        # normal desired velocity: +x at 5 m/s


# =============================================================================
# Unit tests
# =============================================================================

def test_not_trapped_passes_goal_through():
    esc = EscapeController(1, DEV)
    v, escaping = esc.step(0.0, at(), GOAL_V, HeadingMemory(1, DEV), FakeDetector([False]))
    assert not escaping[0] and torch.allclose(v, GOAL_V)


def test_trapped_heads_for_best_remembered_direction():
    mem = HeadingMemory(1, DEV)
    mem.update(0.0, at(), rays([0, 90, 180]), dists([30.0, 6.0, 10.0]))
    esc = EscapeController(1, DEV, speed_frac=0.5)
    v, escaping = esc.step(0.0, at(), GOAL_V, mem, FakeDetector([True], trap_yaw_deg=0))
    assert escaping[0]
    # 0 deg is the most open but it's the blocked goal direction -> excluded; next is 180
    assert abs(abs(heading_deg(v[0])) - 180) < 1e-3, heading_deg(v[0])
    assert abs(v[0].norm().item() - 2.5) < 1e-5                # half the normal speed
    assert abs(v[0, 2].item()) < 1e-9                         # level


def test_empty_memory_falls_back_to_reverse():
    esc = EscapeController(1, DEV)
    v, _ = esc.step(0.0, at(), GOAL_V, HeadingMemory(1, DEV), FakeDetector([True], trap_yaw_deg=30))
    assert abs(heading_deg(v[0]) - (30 - 180)) < 1e-3, heading_deg(v[0])


def test_repicks_when_chosen_heading_is_blocked():
    mem = HeadingMemory(1, DEV)
    mem.update(0.0, at(), rays([90, 180]), dists([6.0, 10.0]))
    det = FakeDetector([True], trap_yaw_deg=0)
    esc = EscapeController(1, DEV, repick_s=0.5, min_free=2.0)
    v, _ = esc.step(0.0, at(), GOAL_V, mem, det)
    assert abs(abs(heading_deg(v[0])) - 180) < 1e-3

    mem.update(0.2, at(), rays([180]), dists([1.0]))          # turned around: it's blocked
    v, _ = esc.step(0.2, at(), GOAL_V, mem, det)
    assert abs(abs(heading_deg(v[0])) - 180) < 1e-3           # too soon to re-pick (0.2 < 0.5 s)

    v, _ = esc.step(0.6, at(), GOAL_V, mem, det)
    assert abs(heading_deg(v[0]) - 90) < 1e-3, heading_deg(v[0])
    assert esc.n_repicks[0].item() == 1


def test_finishes_after_exit_distance():
    det = FakeDetector([True], trap_yaw_deg=0)
    esc = EscapeController(1, DEV, exit_dist=8.0)
    mem = HeadingMemory(1, DEV)
    esc.step(0.0, at(), GOAL_V, mem, det)
    _, escaping = esc.step(1.0, at(x=-7.0), GOAL_V, mem, det)
    assert escaping[0]                                        # 7 m: not yet
    v, escaping = esc.step(2.0, at(x=-8.5), GOAL_V, mem, det)
    assert not escaping[0] and det.cleared[0]
    assert torch.allclose(v, GOAL_V)                          # back to normal
    assert esc.has_last_trap[0] and esc.n_escapes[0].item() == 1


def test_finishes_after_timeout():
    det = FakeDetector([True])
    esc = EscapeController(1, DEV, timeout=8.0)
    mem = HeadingMemory(1, DEV)
    esc.step(0.0, at(), GOAL_V, mem, det)
    _, escaping = esc.step(7.9, at(), GOAL_V, mem, det)
    assert escaping[0]
    _, escaping = esc.step(8.0, at(), GOAL_V, mem, det)
    assert not escaping[0] and det.cleared[0]


def test_batch_independent():
    mem = HeadingMemory(2, DEV)
    det = FakeDetector([False, True], trap_yaw_deg=0)
    esc = EscapeController(2, DEV)
    gv = GOAL_V.repeat(2, 1)
    v, escaping = esc.step(0.0, torch.cat([at(), at()]), gv, mem, det)
    assert not escaping[0] and escaping[1]
    assert torch.allclose(v[0], gv[0])
    assert not torch.allclose(v[1], gv[1])


# =============================================================================
# Closed loop: point-mass drone flying the U-trap
# =============================================================================

WALLS = [(12, -3, 12, 3),      # back wall
         (8, 3, 12, 3),        # left side
         (8, -3, 12, -3)]      # right side
CUP_OPENING_X = 8.0
CAM = [-39 + 78 * i / 16 for i in range(17)]                 # +-39 deg view


def cast(ox, oy, yaw_rad, walls=WALLS):
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


def run_utrap(seconds=15.0, use_layer_b=True, T=0.5, radius=0.5, v_max=5.0):
    """
    A point mass follows the desired velocity with a first-order lag T. It stops
    instead of entering a wall (like the reactive policy hovering at the back wall).
    The camera faces the desired direction. Returns a log of (t, x, y, escaping).
    """
    goal = torch.tensor([[20.0, 0.0, 1.0]])
    mem, det, esc = HeadingMemory(1, DEV), TrapDetector(1, DEV), EscapeController(1, DEV)
    p, v = [0.0, 0.0], [0.0, 0.0]
    head = 0.0
    log = []
    for k in range(int(seconds / DT)):
        t = k * DT
        pos = torch.tensor([[p[0], p[1], 1.0]])

        # camera -> memory
        d = [cast(p[0], p[1], head + math.radians(a)) for a in CAM]
        mem.update(t, pos, rays([math.degrees(head) + a for a in CAM]), dists(d))

        # normal desired velocity: straight at the goal
        g = (goal - pos)[0].clone()
        g[2] = 0.0
        goal_vel = (g / g.norm() * min(g.norm().item(), v_max))[None]

        if use_layer_b:
            det.update(t, pos, goal)
            des, escaping = esc.step(t, pos, goal_vel, mem, det)
            escaping = bool(escaping[0])
        else:
            des, escaping = goal_vel, False
        dx, dy = des[0, 0].item(), des[0, 1].item()
        if dx * dx + dy * dy > 1e-9:
            head = math.atan2(dy, dx)

        # dynamics: lagged velocity, stop at walls
        v[0] += (dx - v[0]) * DT / T
        v[1] += (dy - v[1]) * DT / T
        sp = math.hypot(v[0], v[1])
        if sp > 1e-6:
            if cast(p[0], p[1], math.atan2(v[1], v[0])) < sp * DT + radius:
                v = [0.0, 0.0]
            else:
                p = [p[0] + v[0] * DT, p[1] + v[1] * DT]
        log.append((t, p[0], p[1], escaping))
    return log, esc, det


def inside_cup(x, y):
    return x > CUP_OPENING_X and abs(y) < 3.0


def test_without_layer_b_stuck_in_cup():
    log, _, _ = run_utrap(use_layer_b=False)
    t, x, y, _ = log[-1]
    assert inside_cup(x, y), (x, y)                            # pressed against the back wall


def test_layer_b_escapes_the_cup():
    log, esc, det = run_utrap(use_layer_b=True)
    assert esc.n_escapes[0].item() >= 1

    # find the end of the first escape
    exit_idx = next(i for i in range(1, len(log)) if log[i - 1][3] and not log[i][3])
    t_exit, x, y, _ = log[exit_idx]
    assert not inside_cup(x, y), (x, y)                        # it's out
    assert x < CUP_OPENING_X - 1.0, x                          # clearly out, not at the lip

    # the escape started only after reaching the wall and stalling
    t_start = next(t for (t, _, _, e) in log if e)
    assert t_start > 2.0, t_start


def test_without_step4_it_flies_back_in():
    """Documents the gap step 4 fills: after escaping, aiming at the goal leads back into the cup."""
    log, esc, _ = run_utrap(seconds=15.0, use_layer_b=True)
    exit_idx = next(i for i in range(1, len(log)) if log[i - 1][3] and not log[i][3])
    later = log[exit_idx:]
    assert any(inside_cup(x, y) for (_, x, y, _) in later)


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