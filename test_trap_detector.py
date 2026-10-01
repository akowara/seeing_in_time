"""
Tests for Layer B step 2 (trap detector).

Run:  python test_trap_detector.py      (or: pytest test_trap_detector.py)
Runs on CPU; needs trap_detector.py next to this file.
"""

import math
import torch

from trap_detector import TrapDetector

DEV = 'cpu'
DT = 1 / 15                      # control rate of the sim
GOAL = torch.tensor([[30.0, 0.0, 1.0]])


def at(x=0.0, y=0.0, z=1.0):
    return torch.tensor([[x, y, z]])


def fly(det, xs, ys=None, goal=GOAL, blocked=None, t0=0.0):
    """Feed a trajectory; returns the time of the first trap (or None)."""
    ys = ys or [0.0] * len(xs)
    first = None
    for k, (x, y) in enumerate(zip(xs, ys)):
        t = t0 + k * DT
        b = None if blocked is None else torch.tensor([blocked[k]])
        if det.update(t, at(x, y), goal, b).any() and first is None:
            first = t
    return first


def steps(seconds):
    return int(round(seconds / DT)) + 1


# =============================================================================

def test_steady_progress_never_trapped():
    det = TrapDetector(1, DEV)
    xs = [5.0 * k * DT for k in range(steps(5))]             # 5 m/s toward the goal
    assert fly(det, xs) is None


def test_slow_steady_progress_not_trapped():
    det = TrapDetector(1, DEV)                                # 0.6 m/s -> 1.2 m per 2 s > 1 m
    xs = [0.6 * k * DT for k in range(steps(6))]
    assert fly(det, xs) is None


def test_hovering_trapped_after_window_not_before():
    det = TrapDetector(1, DEV, window=2.0)
    first = fly(det, [10.0] * steps(5))
    assert first is not None
    assert 2.0 - 1e-6 <= first <= 2.0 + 2 * DT, first          # needs a full window first


def test_near_goal_not_trapped():
    det = TrapDetector(1, DEV, near_goal=3.0)
    assert fly(det, [28.5] * steps(5)) is None                # 1.5 m from the goal


def test_sideways_shuffle_at_wall_is_trapped():
    """Sliding back and forth along a wall: moving, but no goal progress."""
    det = TrapDetector(1, DEV)
    n = steps(5)
    ys = [2.0 * math.sin(k * DT * 2.0) for k in range(n)]
    assert fly(det, [11.0] * n, ys) is not None


def test_all_blocked_triggers_even_while_moving():
    det = TrapDetector(1, DEV, blocked_s=1.0)
    n = steps(3)
    xs = [5.0 * k * DT for k in range(n)]                     # good progress
    blocked = [k * DT >= 0.5 for k in range(n)]               # but blocked from 0.5 s
    first = fly(det, xs, blocked=blocked)
    assert first is not None and 1.5 - 1e-6 <= first <= 1.5 + 2 * DT, first
    assert det.reason[0].item() == 2


def test_blocked_must_be_continuous():
    det = TrapDetector(1, DEV, blocked_s=1.0)
    n = steps(3)
    xs = [5.0 * k * DT for k in range(n)]
    blocked = [(k // 5) % 2 == 0 for k in range(n)]           # on/off every 5 steps (~0.33 s)
    assert fly(det, xs, blocked=blocked) is None


def test_records_trap_position_and_blocked_direction():
    det = TrapDetector(1, DEV)
    goal = torch.tensor([[10.0, 10.0, 1.0]])                  # goal at 45 deg
    fly(det, [0.0] * steps(3), goal=goal)
    assert det.trapped[0]
    assert torch.allclose(det.trap_pos[0], at()[0])
    assert abs(math.degrees(det.trap_yaw[0].item()) - 45.0) < 1e-4
    assert det.reason[0].item() == 1


def test_latches_and_reports_once():
    det = TrapDetector(1, DEV)
    news = 0
    for k in range(steps(5)):
        news += int(det.update(k * DT, at(10.0), GOAL).item())
    assert news == 1 and det.trapped[0]


def test_clear_needs_fresh_history():
    det = TrapDetector(1, DEV, window=2.0)
    t = 0.0
    for k in range(steps(3)):                                 # get trapped
        t = k * DT
        det.update(t, at(10.0), GOAL)
    assert det.trapped[0]
    det.clear(torch.tensor([True]), t)
    assert not det.trapped[0]

    first = None
    for k in range(1, steps(4)):                              # keep hovering after clear
        tk = t + k * DT
        if det.update(tk, at(10.0), GOAL).any() and first is None:
            first = tk - t
    assert first is not None and first >= 2.0 - 1e-6, first   # not instantly, only after a new window


def test_batch_independent():
    det = TrapDetector(2, DEV)
    goal = GOAL.repeat(2, 1)
    for k in range(steps(4)):
        pos = torch.tensor([[5.0 * k * DT, 0.0, 1.0],         # drone 0 progresses
                            [10.0, 0.0, 1.0]])                # drone 1 hovers
        det.update(k * DT, pos, goal)
    assert not det.trapped[0] and det.trapped[1]


def test_mini_utrap_trapped_only_after_reaching_wall():
    """Fly in at 5 m/s, stop at the back wall (x = 11) and jitter there."""
    det = TrapDetector(1, DEV, window=2.0)
    xs, ys = [], []
    for k in range(steps(6)):
        t = k * DT
        if t < 2.2:
            xs.append(5.0 * t); ys.append(0.0)                # flying in
        else:
            xs.append(11.0 + 0.1 * math.sin(10 * t))          # pressed against the wall
            ys.append(0.3 * math.sin(3 * t))
    first = fly(det, xs, ys)
    assert first is not None
    assert 2.2 < first <= 2.2 + 2.0 + 3 * DT, first           # within one window of arriving


# =============================================================================
# progress='command': judge progress along the commanded direction
# =============================================================================

def run_cmd(det, positions, cmds):
    first = None
    for k, ((x, y), (cx, cy)) in enumerate(zip(positions, cmds)):
        t = k * DT
        cmd = torch.tensor([[cx, cy, 0.0]])
        if det.update(t, at(x, y), GOAL, cmd_vel=cmd).any() and first is None:
            first = t
    return first


def detour_path(seconds=5.0, speed=2.0):
    """Moving sideways (+y) at 2 m/s: no goal progress, but exactly as commanded."""
    n = steps(seconds)
    pos = [(5.0, speed * k * DT) for k in range(n)]
    cmds = [(0.0, speed)] * n
    return pos, cmds


def test_goal_mode_flags_a_deliberate_detour():
    """Documents the problem: a sideways detour looks like a stall to goal-progress."""
    det = TrapDetector(1, DEV, progress='goal')
    pos, cmds = detour_path()
    assert run_cmd(det, pos, cmds) is not None


def test_command_mode_allows_a_deliberate_detour():
    det = TrapDetector(1, DEV, progress='command')
    pos, cmds = detour_path()
    assert run_cmd(det, pos, cmds) is None


def test_command_mode_still_catches_pressing_into_a_wall():
    det = TrapDetector(1, DEV, progress='command')
    n = steps(5)
    first = run_cmd(det, [(11.0, 0.0)] * n, [(5.0, 0.0)] * n)    # told to go +x, not moving
    assert first is not None and 2.0 - 1e-6 <= first <= 2.0 + 2 * DT, first


def test_command_mode_still_catches_sideways_shuffle():
    det = TrapDetector(1, DEV, progress='command')
    n = steps(5)
    pos = [(11.0, 2.0 * math.sin(k * DT * 2.0)) for k in range(n)]
    assert run_cmd(det, pos, [(5.0, 0.0)] * n) is not None       # told +x, sliding along y


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