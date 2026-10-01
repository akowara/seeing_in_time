"""
Tests for the TTC-space planner, Layer A (steps 1-4).

Run:  python test_ttc_planner.py      (or: pytest test_ttc_planner.py)

The functions under "Planner code" are the versions from our walkthrough.
When they live in your own module, delete that section and import instead, e.g.
    from ttc_planner import lift_ttc, make_candidates, rollout, score, select, select_local
"""

import math
import torch

torch.manual_seed(0)

# Camera for a 36x48 TTC map (placeholder intrinsics; swap in your scaled ones)
H, W = 36, 48
FX, FY, CX, CY = 40.0, 40.0, 24.0, 18.0


# =============================================================================
# Planner code
# =============================================================================

def lift_ttc(ttc, fx, fy, cx, cy, ttc_max=24.0):
    H, W = ttc.shape
    kw = dict(device=ttc.device, dtype=ttc.dtype)
    u, v = torch.meshgrid(torch.arange(W, **kw) + 0.5,
                          torch.arange(H, **kw) + 0.5, indexing='xy')
    xb = (u - cx) / fx
    yb = (v - cy) / fy
    q = torch.stack([ttc * xb, ttc * yb, ttc], dim=-1)
    valid = torch.isfinite(ttc) & (ttc > 0) & (ttc < ttc_max)
    return q, valid


def make_candidates(n_yaw=21, n_pitch=3,          # odd -> includes straight ahead
                    yaw_max=math.radians(60), pitch_max=math.radians(20),
                    speed_scales=(0.5, 0.75, 1.0)):
    yaws = torch.linspace(-yaw_max, yaw_max, n_yaw)
    pitches = torch.linspace(-pitch_max, pitch_max, n_pitch)
    cands = []
    for s in speed_scales:
        for yaw in yaws:
            for pitch in pitches:
                d = torch.stack([torch.sin(yaw) * torch.cos(pitch),
                                 -torch.sin(pitch),
                                 torch.cos(yaw) * torch.cos(pitch)])
                cands.append(s * d)
    return torch.stack(cands)


def rollout(cands, T=0.3, horizon=1.5, dt=0.1):
    """horizon is measured at the current speed; each candidate is rolled out
    for horizon / speed_scale seconds, so all paths look the same distance ahead."""
    t = torch.arange(dt, horizon + 1e-6, dt, device=cands.device, dtype=cands.dtype)
    t = t[None, :] / cands.norm(dim=-1, keepdim=True)          # (N, K)
    v_now = torch.tensor([0.0, 0.0, 1.0], device=cands.device, dtype=cands.dtype)
    tt = t[:, :, None]                                         # (N, K, 1)
    decay = torch.exp(-tt / T)
    vc = cands[:, None, :]
    D = vc * tt + (v_now - vc) * T * (1 - decay)
    V = vc + (v_now - vc) * decay
    return D, V


def min_headway(pts, D, V):
    """Closest time headway (s) of each candidate's path to any point. (N,)"""
    if pts.shape[0] == 0:
        return torch.full((D.shape[0],), float('inf'), dtype=D.dtype)
    dist = torch.cdist(D, pts[None].expand(D.shape[0], -1, -1))   # (N, K, P)
    speed = V.norm(dim=-1, keepdim=True)                           # (N, K, 1)
    return (dist / speed).amin(dim=(1, 2))


def score(q, valid, cands, D, V, goal_dir, t_min=0.5,
          w_c=100.0, w_g=1.0, w_s=0.2):
    pts = q[valid]
    C_coll = torch.clamp(t_min - min_headway(pts, D, V), min=0) ** 2
    d = cands / cands.norm(dim=-1, keepdim=True)
    C_goal = 1 - d @ goal_dir
    v_now = torch.tensor([0.0, 0.0, 1.0], dtype=cands.dtype, device=cands.device)
    C_smooth = (cands - v_now).norm(dim=-1)
    return w_c * C_coll + w_g * C_goal + w_s * C_smooth


def select(cands, J, lam=0.02):
    w = torch.softmax(-(J - J.min()) / lam, dim=0)
    return (w[:, None] * cands).sum(dim=0)


def select_local(cands, J, lam=0.02, max_angle=math.radians(20)):
    """Softmin, but only over candidates within max_angle of the best one.
    Avoids averaging 'go left' and 'go right' into 'go straight'."""
    d = cands / cands.norm(dim=-1, keepdim=True)
    best = d[J.argmin()]
    near = (d @ best) >= math.cos(max_angle)
    Jm = torch.where(near, J, torch.full_like(J, float('inf')))
    w = torch.softmax(-(Jm - J.min()) / lam, dim=0)
    return (w[:, None] * cands).sum(dim=0)


# =============================================================================
# Helpers for building test scenes
# =============================================================================

def pixel_rays():
    """Normalized coords (xb, yb) of every pixel, computed independently of lift_ttc."""
    xb = torch.empty(H, W)
    yb = torch.empty(H, W)
    for r in range(H):
        for c in range(W):
            xb[r, c] = (c + 0.5 - CX) / FX
            yb[r, c] = (r + 0.5 - CY) / FY
    return xb, yb


def pole_scene(col_lo, col_hi, tau_pole=1.5):     # must be inside the horizon
    """A full-height pole at TTC tau_pole in columns [col_lo, col_hi); open elsewhere."""
    ttc = torch.full((H, W), 24.0)          # 24 = clip value = free
    ttc[:, col_lo:col_hi] = tau_pole
    return ttc


GOAL_AHEAD = torch.tensor([0.0, 0.0, 1.0])


# =============================================================================
# Step 1: lift
# =============================================================================

def test_lift_shape_and_centre():
    q, valid = lift_ttc(torch.full((H, W), 2.0), FX, FY, CX, CY)
    assert q.shape == (H, W, 3)
    assert valid.all()
    # centre pixel sits half a pixel off the optical axis
    assert torch.allclose(q[18, 24], torch.tensor([0.5 / FX * 2, 0.5 / FY * 2, 2.0]))


def test_lift_not_transposed():
    """Catches the meshgrid 'ij' vs 'xy' bug: x must vary along columns."""
    q, _ = lift_ttc(torch.ones(H, W), FX, FY, CX, CY)
    assert q[0, -1, 0] > 0 > q[0, 0, 0]      # right column has positive x
    assert q[-1, 0, 1] > 0 > q[0, 0, 1]      # bottom row has positive y (y down)


def test_lift_matches_metric_depth():
    """q * forward_speed must equal the true 3D point, for any speed."""
    xb, yb = pixel_rays()
    Z = 2.0 + 30.0 * torch.rand(H, W)                      # true depth (m)
    P = torch.stack([Z * xb, Z * yb, Z], dim=-1)           # true 3D points (m)
    for vz in (3.0, 8.0, 15.0):
        ttc = Z / vz
        q, _ = lift_ttc(ttc, FX, FY, CX, CY, ttc_max=1e9)
        assert torch.allclose(q * vz, P, rtol=1e-5, atol=1e-4), f"speed {vz}"


def test_lift_is_scale_free():
    """A big world at high speed and a small world at low speed give the same q."""
    xb, yb = pixel_rays()
    Z = 2.0 + 30.0 * torch.rand(H, W)
    q_big, _ = lift_ttc((3.0 * Z) / (3.0 * 10.0), FX, FY, CX, CY)
    q_small, _ = lift_ttc(Z / 10.0, FX, FY, CX, CY)
    assert torch.allclose(q_big, q_small, rtol=1e-5)


def test_lift_valid_mask():
    ttc = torch.full((H, W), 5.0)
    ttc[0, 0] = 24.0            # at clip -> free/unknown
    ttc[0, 1] = float('inf')
    ttc[0, 2] = 0.0
    ttc[0, 3] = -1.0
    _, valid = lift_ttc(ttc, FX, FY, CX, CY)
    assert not valid[0, :4].any()
    assert valid[0, 4:].all() and valid[1:].all()


# =============================================================================
# Step 2: candidates
# =============================================================================

def test_candidates_shape_and_speeds():
    c = make_candidates()
    assert c.shape == (21 * 3 * 3, 3)
    norms = torch.unique(torch.round(c.norm(dim=-1), decimals=5))
    assert torch.allclose(norms, torch.tensor([0.5, 0.75, 1.0]))


def test_candidates_cone_limits():
    d = make_candidates()
    d = d / d.norm(dim=-1, keepdim=True)
    yaw = torch.atan2(d[:, 0], d[:, 2])
    pitch = torch.asin(-d[:, 1])
    assert torch.isclose(yaw.abs().max(), torch.tensor(math.radians(60)), atol=1e-5)
    assert torch.isclose(pitch.abs().max(), torch.tensor(math.radians(20)), atol=1e-5)
    assert (d[:, 2] > 0).all()                       # everything goes forward


def test_candidates_include_straight_ahead():
    c = make_candidates()
    assert any(torch.allclose(v, torch.tensor([0.0, 0.0, 1.0]), atol=1e-6) for v in c)


def test_candidates_pitch_up_is_negative_y():
    c = make_candidates(n_yaw=1, n_pitch=2, yaw_max=0.0, speed_scales=(1.0,))
    # pitches are [-20deg, +20deg]; +20deg (up) must have y < 0
    assert c[1, 1] < 0 < c[0, 1]


# =============================================================================
# Step 3: rollout
# =============================================================================

def test_rollout_shapes():
    D, V = rollout(make_candidates())
    assert D.shape == V.shape == (189, 15, 3)


def test_rollout_same_distance_for_all_speeds():
    """Straight at half speed must reach as far as straight at full speed."""
    D, _ = rollout(torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.5]]))
    assert D[1, -1, 2] >= D[0, -1, 2] - 1e-5


def test_rollout_straight_command_is_straight_line():
    D, V = rollout(torch.tensor([[0.0, 0.0, 1.0]]))
    t = torch.arange(0.1, 1.5 + 1e-6, 0.1)
    assert torch.allclose(D[0, :, 2], t, atol=1e-5)
    assert torch.allclose(D[0, :, :2], torch.zeros(15, 2), atol=1e-6)
    assert torch.allclose(V[0], torch.tensor([0.0, 0.0, 1.0]).expand(15, 3))


def test_rollout_position_is_integral_of_velocity():
    """Closed-form D must match numerically integrating V."""
    cands = make_candidates()
    dt = 1e-3
    D, V = rollout(cands, dt=dt)
    dt_i = dt / cands.norm(dim=-1)[:, None, None]    # each candidate's own step
    v0 = torch.tensor([0.0, 0.0, 1.0]).expand(cands.shape[0], 1, 3)
    Vfull = torch.cat([v0, V], dim=1)
    D_num = torch.cumsum(0.5 * (Vfull[:, 1:] + Vfull[:, :-1]) * dt_i, dim=1)
    assert torch.allclose(D, D_num, atol=1e-3)


def test_rollout_velocity_converges_to_command():
    cands = make_candidates()
    _, V = rollout(cands, T=0.1, horizon=1.5)        # 15 time constants
    assert torch.allclose(V[:, -1], cands, atol=1e-5)


def test_forward_model_matches_brief_closed_form():
    """Flying straight for t seconds: tau' = tau - t, and pixels expand
    from the centre by tau / (tau - t), exactly as in the brief."""
    xb, yb = pixel_rays()
    ttc = 3.0 + 10.0 * torch.rand(H, W)
    q, _ = lift_ttc(ttc, FX, FY, CX, CY)
    D, _ = rollout(torch.tensor([[0.0, 0.0, 1.0]]))
    t = 1.0
    q_new = q - D[0, 9]                               # index 9 -> t = 1.0 s
    assert torch.allclose(q_new[..., 2], ttc - t, atol=1e-5)
    xb_new = q_new[..., 0] / q_new[..., 2]
    assert torch.allclose(xb_new, xb * ttc / (ttc - t), atol=1e-5)


# =============================================================================
# Step 4: score + select
# =============================================================================

def plan(ttc, goal_dir=GOAL_AHEAD, selector=select):
    q, valid = lift_ttc(ttc, FX, FY, CX, CY)
    cands = make_candidates()
    D, V = rollout(cands)
    J = score(q, valid, cands, D, V, goal_dir)
    return selector(cands, J), cands, J, q, valid


def headway_of(v, q, valid):
    D, V = rollout(v[None])
    return min_headway(q[valid], D, V).item()


def test_open_scene_goes_to_goal():
    """Nothing in view: fly toward the goal at full speed."""
    v, *_ = plan(torch.full((H, W), 24.0))
    assert torch.allclose(v, GOAL_AHEAD, atol=0.05)


def test_open_scene_follows_offset_goal():
    goal = torch.tensor([math.sin(math.radians(30)), 0.0, math.cos(math.radians(30))])
    v, *_ = plan(torch.full((H, W), 24.0), goal_dir=goal)
    yaw = math.degrees(math.atan2(v[0], v[2]))
    assert 15 < yaw < 45, yaw


def test_straight_into_pole_is_penalized():
    ttc = pole_scene(20, 28)
    _, cands, J, *_ = plan(ttc)
    straightest = (cands / cands.norm(dim=-1, keepdim=True))[:, 2].argmax()
    assert J[straightest] > J.min() + 1.0


def test_offset_pole_steers_to_open_side():
    """Pole just left of centre: go right, with a safe margin."""
    v, _, _, q, valid = plan(pole_scene(16, 24))
    assert v[0] > 0.1, f"expected to steer right, got {v.tolist()}"
    assert headway_of(v, q, valid) >= 0.5


def test_symmetric_pole_averaging_trap():
    """Pole dead centre: left and right tie. Plain softmin averages them into
    'straight ahead' (into the pole); select_local picks a side."""
    ttc = pole_scene(20, 28)
    v_avg, _, _, q, valid = plan(ttc, selector=select)
    v_loc, *_ = plan(ttc, selector=select_local)

    print(f"\n    plain softmin -> x = {v_avg[0]:+.3f}, headway {headway_of(v_avg, q, valid):.2f} s")
    print(f"    select_local  -> x = {v_loc[0]:+.3f}, headway {headway_of(v_loc, q, valid):.2f} s")

    assert abs(v_avg[0]) < 0.05                      # averaged into straight ahead...
    assert headway_of(v_avg, q, valid) < 0.5         # ...which is unsafe (the trap)
    assert abs(v_loc[0]) > 0.1                       # the fix commits to a side...
    assert headway_of(v_loc, q, valid) >= 0.5        # ...and keeps the margin


# =============================================================================

if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}  {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")