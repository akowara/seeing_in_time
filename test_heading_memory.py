"""
Tests for Layer B step 1 (heading memory).

Run:  python test_heading_memory.py      (or: pytest test_heading_memory.py)
Runs on CPU; needs heading_memory.py next to this file.
"""

import math
import torch

from heading_memory import HeadingMemory

DEV = 'cpu'


def rays(yaws_deg, elevs_deg=None):
    """(1, P, 3) unit rays at the given yaws (and elevations), world frame."""
    elevs_deg = elevs_deg or [0.0] * len(yaws_deg)
    out = []
    for y, e in zip(yaws_deg, elevs_deg):
        y, e = math.radians(y), math.radians(e)
        out.append([math.cos(e) * math.cos(y), math.cos(e) * math.sin(y), math.sin(e)])
    return torch.tensor([out], dtype=torch.float32)


def dists(values):
    return torch.tensor([values], dtype=torch.float32)


def at(x=0.0, y=0.0, z=1.0):
    return torch.tensor([[x, y, z]], dtype=torch.float32)


def deg(t):
    return math.degrees(float(t))


def yaw(d):
    return torch.tensor([math.radians(d)])


# =============================================================================
# Camera updates
# =============================================================================

def test_single_ray():
    m = HeadingMemory(1, DEV)
    m.update(0.0, at(), rays([0]), dists([5.0]))
    assert m.seen.sum() == 1
    assert abs(m.free_toward(yaw(0)).item() - 5.0) < 1e-6


def test_closest_obstacle_per_bin():
    m = HeadingMemory(1, DEV)
    m.update(0.0, at(), rays([1, 3]), dists([5.0, 2.0]))      # same 10-degree bin
    assert abs(m.free_toward(yaw(2)).item() - 2.0) < 1e-6


def test_no_hit_stores_cap():
    m = HeadingMemory(1, DEV, d_cap=30.0)
    m.update(0.0, at(), rays([0]), dists([float('inf')]))
    assert abs(m.free_toward(yaw(0)).item() - 30.0) < 1e-6


def test_out_of_band_rays_ignored():
    """A ray looking 30 deg down at the floor must not mark its direction as blocked."""
    m = HeadingMemory(1, DEV, elev_band_deg=10.0)
    m.update(0.0, at(), rays([0], [-30.0]), dists([1.0]))
    assert m.seen.sum() == 0


def test_memory_persists_out_of_view():
    m = HeadingMemory(1, DEV)
    m.update(0.0, at(), rays([0]), dists([5.0]))
    m.update(1.0, at(), rays([180]), dists([4.0]))            # camera turned around
    assert abs(m.free_toward(yaw(0)).item() - 5.0) < 1e-6    # still remembered
    assert abs(m.age(1.0)[0, m.yaw_to_bin(yaw(0))].item() - 1.0) < 1e-6


def test_reobservation_overwrites():
    m = HeadingMemory(1, DEV)
    m.update(0.0, at(), rays([0]), dists([20.0]))
    m.update(1.0, at(), rays([0]), dists([2.0]))             # a wall appeared
    assert abs(m.free_toward(yaw(0)).item() - 2.0) < 1e-6


def test_unseen_is_nan():
    m = HeadingMemory(1, DEV)
    assert math.isnan(m.free_toward(yaw(90)).item())


# =============================================================================
# Trail ("the way back is free")
# =============================================================================

def test_trail_marks_way_back_free():
    m = HeadingMemory(1, DEV, trail_s=3.0)
    for k in range(11):                                       # fly +x, 0 -> 3 m in 1 s
        m.update(k * 0.1, at(x=0.3 * k))
    back = m.free_toward(yaw(180)).item()
    assert abs(back - 3.0) < 1e-4, back


def test_trail_uses_position_from_trail_s_ago():
    m = HeadingMemory(1, DEV, trail_s=1.0)
    for k in range(31):                                       # fly +x at 1 m/s for 3 s
        m.update(k * 0.1, at(x=0.1 * k))
    back = m.free_toward(yaw(180)).item()
    assert 0.9 <= back <= 1.1, back                           # ~1 s worth of trail, not 3 m


def test_camera_overrides_trail():
    m = HeadingMemory(1, DEV)
    for k in range(11):
        m.update(k * 0.1, at(x=0.3 * k))
    m.update(1.1, at(x=3.0), rays([180]), dists([0.8]))       # something moved in behind
    assert abs(m.free_toward(yaw(180)).item() - 0.8) < 1e-6


# =============================================================================
# Queries
# =============================================================================

def test_best_heading_and_avoid():
    m = HeadingMemory(1, DEV)
    m.update(0.0, at(), rays([0, 90]), dists([2.0, 20.0]))
    y, f = m.best_heading(0.0)
    assert abs(deg(y[0]) - 90) <= 5 and abs(f.item() - 20.0) < 1e-6
    y, f = m.best_heading(0.0, avoid_yaw=yaw(90))             # 90 excluded -> next best
    assert abs(deg(y[0]) - 0) <= 5 and abs(f.item() - 2.0) < 1e-6


def test_best_heading_max_age():
    m = HeadingMemory(1, DEV)
    m.update(0.0, at(), rays([90]), dists([20.0]))
    m.update(5.0, at(), rays([0]), dists([3.0]))
    y, _ = m.best_heading(5.0, max_age=2.0)                   # the 90-deg reading is 5 s old
    assert abs(deg(y[0]) - 0) <= 5


def test_nothing_usable_returns_minus_one():
    m = HeadingMemory(1, DEV)
    _, f = m.best_heading(0.0)
    assert f.item() == -1.0


def test_batch_independent():
    m = HeadingMemory(2, DEV)
    r = torch.cat([rays([0]), rays([0])])
    m.update(0.0, torch.cat([at(), at()]), r, torch.tensor([[5.0], [1.0]]))
    v = m.free_toward(torch.tensor([0.0, 0.0]))
    assert abs(v[0] - 5.0) < 1e-6 and abs(v[1] - 1.0) < 1e-6


# =============================================================================
# Mini U-trap: fly into a cup, camera forward; memory should point back out
# =============================================================================

def cast(ox, oy, yaw_rad, walls, max_d=1e9):
    """2D ray cast against wall segments; distance to the first hit (inf if none)."""
    dx, dy = math.cos(yaw_rad), math.sin(yaw_rad)
    best = float('inf')
    for (ax, ay, bx, by) in walls:
        ex, ey = bx - ax, by - ay
        rx, ry = ax - ox, ay - oy
        det = -dx * ey + ex * dy
        if abs(det) < 1e-12:
            continue
        s = (-rx * ey + ex * ry) / det          # distance along the ray
        u = (dx * ry - dy * rx) / det           # position along the wall
        if s > 1e-9 and 0.0 <= u <= 1.0:
            best = min(best, s)
    return best


def test_utrap_points_back_out():
    walls = [(12, -3, 12, 3),      # back wall
             (8, 3, 12, 3),        # left side
             (8, -3, 12, -3)]      # right side
    cam = [-39 + 78 * i / 16 for i in range(17)]              # +-39 deg view, facing +x
    m = HeadingMemory(1, DEV)
    t, x = 0.0, 0.0
    while x <= 11.0 + 1e-9:                                   # fly in at 5 m/s
        d = [cast(x, 0.0, math.radians(a), walls) for a in cam]
        m.update(t, at(x=x), rays(cam), dists(d))
        t += 0.1
        x += 0.5

    assert m.free_toward(yaw(0)).item() < 2.0                 # straight ahead: the back wall
    y, f = m.best_heading(t)
    assert abs(abs(deg(y[0])) - 180) <= 15, deg(y[0])         # best way out: back the way we came
    assert f.item() > 8.0                                     # and it's known to be long


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