"""
TTC-space lookahead planner, Layer A.

Pipeline: lift -> candidates -> rollout -> score -> select

All quantities are time-normalized (divided by the current forward speed),
so no metric depth or speed is needed. Camera frame: x right, y down, z forward.
"""

import math
import torch


def lift_ttc(ttc, fx, fy, cx, cy, ttc_max=24.0):
    """
    Lift a TTC map to time-normalized 3D points.

    ttc: (H, W) TTC map in seconds
    fx, fy, cx, cy: intrinsics scaled to the TTC map's resolution
    Returns q (H, W, 3) in seconds, and a valid mask (H, W).
    Pixels at or beyond ttc_max are treated as free/unknown.
    """
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
    """Returns (N, 3) candidate velocities in time-normalized units."""
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
    """
    First-order rollout of each candidate.

    horizon is measured at the current speed; each candidate is rolled out
    for horizon / speed_scale seconds, so all paths look the same distance ahead.
    T is the policy's velocity-tracking time constant (placeholder until measured).

    Returns D: (N, K, 3) positions, V: (N, K, 3) velocities.
    """
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
        return torch.full((D.shape[0],), float('inf'), dtype=D.dtype, device=D.device)
    dist = torch.cdist(D, pts[None].expand(D.shape[0], -1, -1))   # (N, K, P)
    speed = V.norm(dim=-1, keepdim=True)                           # (N, K, 1)
    return (dist / speed).amin(dim=(1, 2))


def score(q, valid, cands, D, V, goal_dir, t_min=0.5,
          w_c=100.0, w_g=1.0, w_s=0.2):
    """
    Cost per candidate: collision + goal alignment + smoothness. (N,)
    C_unseen and C_mem from the brief are not included yet.
    """
    pts = q[valid]
    C_coll = torch.clamp(t_min - min_headway(pts, D, V), min=0) ** 2
    d = cands / cands.norm(dim=-1, keepdim=True)
    C_goal = 1 - d @ goal_dir
    v_now = torch.tensor([0.0, 0.0, 1.0], dtype=cands.dtype, device=cands.device)
    C_smooth = (cands - v_now).norm(dim=-1)
    return w_c * C_coll + w_g * C_goal + w_s * C_smooth


def select(cands, J, lam=0.02):
    """MPPI-style softmin over all candidates."""
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


def plan(ttc, fx, fy, cx, cy, goal_dir, selector=select_local):
    """One full planning step: TTC map in, desired velocity (3,) out."""
    q, valid = lift_ttc(ttc, fx, fy, cx, cy)
    cands = make_candidates()
    D, V = rollout(cands)
    J = score(q, valid, cands, D, V, goal_dir)
    return selector(cands, J)